# process_gsm8k.py
from datasets import load_dataset
from openai import OpenAI
import time
import re
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import traceback
import argparse
from openai import (
    APIError,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError
)

def exponential_backoff(retry_count, base_wait=1, max_wait=60):
    """Calculate wait time with exponential backoff and jitter"""
    import random
    wait = min(max_wait, base_wait * (2 ** retry_count))
    jitter = random.uniform(0, 0.1 * wait)  # Add 0-10% jitter
    return wait + jitter

def extract_answer(response_text):
    if not response_text:
        return None
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    match = answer_pattern.search(response_text)
    if match:
        return match.group(1).strip()
    return None

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Checkpoint file {checkpoint_file} is corrupted. Creating backup and starting fresh.")
            import shutil
            backup_file = f"{checkpoint_file}.backup.{int(time.time())}"
            shutil.copy2(checkpoint_file, backup_file)
            return []
    return []

def save_checkpoint(results, checkpoint_file):
    temp_file = f"{checkpoint_file}.temp"
    try:
        with open(temp_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if os.path.exists(checkpoint_file):
            os.replace(temp_file, checkpoint_file)
        else:
            os.rename(temp_file, checkpoint_file)
        
        print(f"Checkpoint saved: {len(results)} results")
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")
        traceback.print_exc()

def process_item(idx, dataset, prompt_template, client, max_retries=3):
    input_text = prompt_template.format(dataset['train']['question'][idx], dataset['train']['answer'][idx])
    
    retry_count = 0
    while retry_count <= max_retries:
        try:
            response = client.responses.create(
                model="gpt-4.1-nano",
                input=input_text
            )
            
            answer = extract_answer(response.output_text)
            return {
                'index': idx,
                'question': dataset['train']['question'][idx],
                'expected_answer': dataset['train']['answer'][idx],
                'model_response': response.output_text,
                'extracted_answer': answer,
                'is_correct': answer == dataset['train']['answer'][idx] if answer else False
            }
            
        except RateLimitError as e:
            retry_count += 1
            wait_time = exponential_backoff(retry_count)
            print(f"Rate limit exceeded for item {idx}. Waiting {wait_time:.2f}s before retry {retry_count}/{max_retries}")
            time.sleep(wait_time)
            
            if retry_count > max_retries:
                return {
                    'index': idx,
                    'error': f"Rate limit error after {max_retries} retries: {str(e)}"
                }
                
        except APITimeoutError as e:
            retry_count += 1
            wait_time = exponential_backoff(retry_count, base_wait=2)
            print(f"Timeout for item {idx}. Waiting {wait_time:.2f}s before retry {retry_count}/{max_retries}")
            time.sleep(wait_time)
            
            if retry_count > max_retries:
                return {
                    'index': idx,
                    'error': f"Timeout error after {max_retries} retries: {str(e)}"
                }
                
        except APIConnectionError as e:
            retry_count += 1
            wait_time = exponential_backoff(retry_count, base_wait=5)
            print(f"Connection error for item {idx}. Waiting {wait_time:.2f}s before retry {retry_count}/{max_retries}")
            time.sleep(wait_time)
            
            if retry_count > max_retries:
                return {
                    'index': idx,
                    'error': f"Connection error after {max_retries} retries: {str(e)}"
                }
                
        except InternalServerError as e:
            retry_count += 1
            wait_time = exponential_backoff(retry_count, base_wait=5)
            print(f"Server error for item {idx}. Waiting {wait_time:.2f}s before retry {retry_count}/{max_retries}")
            time.sleep(wait_time)
            
            if retry_count > max_retries:
                return {
                    'index': idx,
                    'error': f"Server error after {max_retries} retries: {str(e)}"
                }
                
        except (AuthenticationError, PermissionDeniedError) as e:
            print(f"Authentication/Permission error for item {idx}: {str(e)}")
            return {
                'index': idx,
                'error': f"Authentication error: {str(e)}"
            }
            
        except BadRequestError as e:
            print(f"Bad request error for item {idx}: {str(e)}")
            return {
                'index': idx,
                'error': f"Bad request error: {str(e)}"
            }
            
        except Exception as e:
            retry_count += 1
            wait_time = exponential_backoff(retry_count)
            print(f"Unexpected error for item {idx}: {str(e)}")
            traceback.print_exc()
            time.sleep(wait_time)
            
            if retry_count > max_retries:
                return {
                    'index': idx,
                    'error': f"Unexpected error after {max_retries} retries: {str(e)}"
                }

def process_in_parallel(dataset, prompt_template, client, start_idx, end_idx, max_workers=10, min_workers=2,
                       checkpoint_file='gsm8k_checkpoint.json',
                       checkpoint_frequency=25, batch_size=500):
    results = load_checkpoint(checkpoint_file)
    
    processed_indices = {r['index'] for r in results}
    
    all_indices = list(range(start_idx, min(end_idx, len(dataset['train']))))
    indices_to_process = [idx for idx in all_indices if idx not in processed_indices]
    
    print(f"Total items: {len(all_indices)}")
    print(f"Already processed: {len(processed_indices)}")
    print(f"Remaining to process: {len(indices_to_process)}")
    
    if not indices_to_process:
        print("All items already processed!")
        return results
    
    num_batches = (len(indices_to_process) + batch_size - 1) // batch_size
    
    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, len(indices_to_process))
        batch_indices = indices_to_process[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_num+1}/{num_batches}: {len(batch_indices)} items")
        
        current_workers = max_workers
        error_count = 0
        
        with ThreadPoolExecutor(max_workers=current_workers) as executor:
            future_to_idx = {executor.submit(
                process_item, idx, dataset, prompt_template, client): idx for idx in batch_indices}
            
            completed = 0
            error_window = []
            
            for future in tqdm(as_completed(future_to_idx), total=len(batch_indices), 
                              desc=f"Batch {batch_num+1}/{num_batches}"):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if 'error' in result:
                        error_count += 1
                        error_window.append(time.time())
                        
                        error_window = [t for t in error_window if time.time() - t < 60]
                        
                        if len(error_window) >= 5 and current_workers > min_workers:
                            current_workers = max(current_workers - 1, min_workers)
                            print(f"\nHigh error rate detected. Reducing workers to {current_workers}")
                    else:
                        if len(error_window) < 2 and current_workers < max_workers and completed > 50:
                            current_workers = min(current_workers + 1, max_workers)
                    
                    if completed % checkpoint_frequency == 0:
                        save_checkpoint(results, checkpoint_file)
                    
                except Exception as e:
                    print(f"Critical error with item {idx}: {str(e)}")
                    traceback.print_exc()
                    results.append({'index': idx, 'error': str(e)})
            
            save_checkpoint(results, checkpoint_file)
            
            print(f"Batch {batch_num+1} complete. Errors: {error_count}/{len(batch_indices)} ({error_count/len(batch_indices)*100:.2f}%)")
            
            if error_count > len(batch_indices) * 0.3:
                pause_time = 60
                print(f"High error rate detected. Pausing for {pause_time}s before next batch")
                time.sleep(pause_time)
    
    processed_indices = {r['index'] for r in results}
    missed_indices = [idx for idx in all_indices if idx not in processed_indices]
    
    if missed_indices:
        print(f"\nSome indices were missed ({len(missed_indices)}). Processing these sequentially...")
        for idx in tqdm(missed_indices, desc="Processing missed items"):
            time.sleep(1)
            result = process_item(idx, dataset, prompt_template, client, max_retries=5)
            results.append(result)
            save_checkpoint(results, checkpoint_file)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Process GSM8K dataset with OpenAI API')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name', default="user074/concat_cleaned_gsm8k_5")
    parser.add_argument('--start_idx', type=int, default=0, help='Starting index in dataset')
    parser.add_argument('--end_idx', type=int, default=-1, help='Ending index in dataset (-1 for all)')
    parser.add_argument('--max_workers', type=int, default=5, help='Maximum parallel workers')
    parser.add_argument('--min_workers', type=int, default=2, help='Minimum parallel workers')
    parser.add_argument('--checkpoint_file', type=str, default='gsm8k_checkpoint.json', help='Checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for processing')
    parser.add_argument('--checkpoint_frequency', type=int, default=25, help='How often to save checkpoints')

    args = parser.parse_args()
    
    # Initialize client with provided API key
    client = OpenAI(api_key=args.api_key)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name)
    
    # Define prompt template
    prompt_template = '''Here is a question and some proposed answers. You need to evaluate each answers individually, check whether each answer directly addresses the original question, assess the correctness of each answer based on logical reasoning, calculations, and accuracy relative to the question. After thorough evaluation, identify one correct answer. If the correct answer is not in the provided proposed answers, the Assistant will combine the correct partial responses to proposed answers and provide the correct answer. Make the reasoning process concise and to the point. The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively,i.e., <think>reasoning process here</think> <answer>answer here</answer>. Here is the Question and proposed answers: {}. 
    Here is the labeled answer with answer tags '<answer>{}</answer>', and most likely the correct answer. If none of the proposed answers is correct, come up with the reasoning process to arrive at the actual correct final answer. **Attention: Do not ever mention the answer is given in the response, but you need to justify it from the problem and proposed answers!**
    '''
    
    # Set end index to dataset size if -1 is provided
    end_idx = len(dataset['train']) if args.end_idx == -1 else args.end_idx
    
    print(f"Processing GSM8K dataset from index {args.start_idx} to {end_idx}")
    print(f"Using {args.max_workers} parallel workers (min: {args.min_workers})")
    print(f"Batch size: {args.batch_size}, Checkpoint frequency: {args.checkpoint_frequency}")
    print(f"Checkpoint file: {args.checkpoint_file}")
    
    # Process dataset
    results = process_in_parallel(
        dataset,
        prompt_template,
        client,
        args.start_idx,
        end_idx,
        max_workers=args.max_workers,
        min_workers=args.min_workers,
        checkpoint_file=args.checkpoint_file,
        checkpoint_frequency=args.checkpoint_frequency,
        batch_size=args.batch_size
    )
    
    # Analyze results
    total_with_answer = sum(1 for r in results if 'is_correct' in r)
    correct_count = sum(1 for r in results if 'is_correct' in r and r['is_correct'])
    error_count = sum(1 for r in results if 'error' in r)
    
    # Categorize errors
    error_types = {}
    for r in results:
        if 'error' in r:
            error_text = r['error']
            for error_type in ["Rate limit", "Timeout", "Connection", "Server", 
                              "Authentication", "Bad request", "Unexpected"]:
                if error_type.lower() in error_text.lower():
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    break
    
    print(f"\nFinal Results:")
    print(f"Processed {len(results)} items")
    print(f"Items with answers: {total_with_answer}")
    print(f"Correct answers: {correct_count} ({correct_count/total_with_answer*100:.2f}% of answered items)")
    print(f"Errors: {error_count} ({error_count/len(results)*100:.2f}%)")
    print("\nError breakdown:")
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count} ({count/error_count*100:.2f}% of errors)")
    
    # Save final results
    print("\nSaving final results...")
    results_file = 'gsm8k_final_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save correct results
    correct_results = [r for r in results if 'is_correct' in r and r['is_correct']]
    correct_file = 'gsm8k_correct_results.json'
    with open(correct_file, 'w') as f:
        json.dump(correct_results, f, indent=2)
    
    print(f"Results saved to {results_file} and {correct_file}")
    print("Processing complete!")

if __name__ == "__main__":
    main()