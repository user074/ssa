import os
import json
import re
import torch
import argparse
from tqdm import tqdm
import sys
import numpy as np
from collections import Counter
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch.nn.functional as F
import time

# Add path for the grader module
sys.path.append('prm800k/prm800k')
from grading import grader, math_normalize

###########################################################################
# Utility functions (reused from original script)
###########################################################################

def normalize_answer(ans):
    """Normalize a generated answer string."""
    if ans is None:
        return None
    if not isinstance(ans, str):
        ans = str(ans)
    ans = math_normalize.normalize_answer(ans)
    ans = ans.strip()
    if ans.startswith('$') and ans.endswith('$'):
        ans = ans[1:-1].strip()
    match = re.fullmatch(r'\\boxed\{(.*)\}', ans)
    if match:
        ans = match.group(1).strip()
    return ans

def clean_generated_text(text):
    """Remove spurious text from generated output."""
    if 'assistant\n' in text:
        if 'is:' in text.split('assistant\n')[1]:
            answer = text.split('assistant\n')[1].split("is:")[-1]
            if '\\' in answer:
                return answer.strip()
            words = answer.split()
            if len(words) > 1:
                try:
                    float(words[0])
                    return words[0]
                except:
                    return answer.strip()
            return answer.strip()
        elif 'boxed{' in text.split('assistant\n')[1]:
            answer = text.split('assistant\n')[1].split('boxed{')[1]
            answer = r'\(\boxed{' + answer
            return answer.strip()
        else:
            return None
    else:
        return None

def extract_answer_from_text(text):
    """Remove spurious text from generated output."""
    if 'is:' in text:
        answer = text.split("is:")[-1]
        if '\\' in answer:
            return answer.strip()
        words = answer.split()
        if len(words) > 1:
            try:
                float(words[0])
                return words[0]
            except:
                return answer.strip()
        return answer.strip()
    elif 'boxed{' in text:
        answer = text.split('boxed{')[1]
        answer = r'\(\boxed{' + answer
        return answer.strip()
    else:
        return None

def compare_with_ground_truth(predicted_answers, ground_truth_answers):
    """Compare predicted answers with ground truth."""
    correct_count = 0
    for pred, gt in tqdm(zip(predicted_answers, ground_truth_answers),
                         total=len(ground_truth_answers),
                         desc="Grading Answers"):
        if grader.grade_answer(pred, gt):
            correct_count += 1
    accuracy = correct_count / len(ground_truth_answers)
    return accuracy

def load_json_answers(directory, num_files=None):
    """Load generated answers from JSON files."""
    json_files = [file for file in os.listdir(directory) if file.endswith(".json")]
    if num_files is not None:
        json_files = json_files[:num_files]
    all_answers = []
    for file in json_files:
        file_path = os.path.join(directory, file)
        with open(file_path, "r") as f:
            answers = json.load(f)
            all_answers.append(answers)
    return all_answers

def remove_system_prompt(answer):
    """Remove system prompt from answer."""
    if 'assistant\n' not in answer:
        return answer.strip()
    return answer.split('assistant\n')[1].strip()

def extract_question(answer):
    """Extract question from answer."""
    if 'user\n' not in answer:
        return ""
    return answer.split('user\n')[1].split('assistant\n')[0].strip()

def normalize_gsm8k_format(answer):
    """Normalize GSM8K answer format."""
    return answer.replace('The Answer is:', '####')

def batch_clean_generated_answers(one_answer):
    """Clean a batch of generated answers."""
    cleaned_answers = []
    questions = []
    for answer in one_answer:
        question = extract_question(answer)
        questions.append(question)
        answer = remove_system_prompt(answer)
        # answer = normalize_gsm8k_format(answer)
        cleaned_answers.append(answer)
    return cleaned_answers, questions

def clean_all_answers(all_answers):
    """Clean all generated answers."""
    cleaned_answers = []
    for answer in all_answers:
        extracted_answers, questions = batch_clean_generated_answers(answer)
        cleaned_answers.append(extracted_answers)
    return cleaned_answers, questions

def get_gsm8k_answers(gsm8k_dataset):
    """Extract answers from GSM8K dataset."""
    gsm8k_answers = [x.split('####')[1].strip() for x in gsm8k_dataset['answer']]
    return gsm8k_answers

def get_dataset_and_answers(dataset_name, dataset_path, question_key, split='test'):
    """Load dataset and extract answers based on dataset type."""
    if dataset_name == 'gsm8k':
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        ground_truth_answers = get_gsm8k_answers(dataset)
    elif dataset_name == 'math':
        dataset = load_dataset("user074/prm800k_MATH_splits", split=split)
        ground_truth_answers = dataset['answer']
    elif dataset_name == 'aime24':
        dataset = load_dataset("Maxwell-Jia/AIME_2024", split='train')
        ground_truth_answers = dataset['Answer']
        ground_truth_answers = [str(x) for x in ground_truth_answers]
    elif dataset_name == 'aime25':
        dataset = load_dataset("yentinglin/aime_2025", split='train')
        ground_truth_answers = dataset['answer']
        ground_truth_answers = [str(x) for x in ground_truth_answers]
    elif dataset_name == 'amc23':
        dataset = load_dataset("zwhe99/amc23", split='test')
        ground_truth_answers = dataset['answer']
        ground_truth_answers = [str(x) for x in ground_truth_answers]
    elif dataset_name == 'olympiad':
        dataset = load_dataset("Hothan/OlympiadBench", 'OE_TO_maths_en_COMP', split='train')
        ground_truth_answers = dataset['final_answer']
        ground_truth_answers = [item for sublist in ground_truth_answers for item in sublist]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return dataset, ground_truth_answers

###########################################################################
# PRM-specific functions
###########################################################################

def split_into_sentences(text):
    """Split text into sentences for PRM evaluation."""
    # Split on double newlines, but handle empty text
    if not text or text.strip() == "":
        return []
        
    # Split on double newlines
    sentences = re.split(r'\n\n', text)
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # If splitting didn't work (no double newlines), use the whole text as one sentence
    if not sentences and text.strip():
        sentences = [text.strip()]
        
    return sentences

def make_step_rewards(logits, token_masks):
    """Calculate step rewards from model logits and token masks."""
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

def calculate_aggregate_score(step_rewards):
    """Calculate aggregate score from step rewards."""
    # Use product of scores as the aggregate
    if not step_rewards or len(step_rewards) == 0:
        return 0.0
    
    product_score = 1.0
    for score in step_rewards:
        product_score *= score
    
    return product_score

###########################################################################
# Model type identification functions
###########################################################################

def is_shepherd_model(model_name):
    """Check if the model is a Shepherd PRM model."""
    return 'shepherd' in model_name.lower()

def is_orm_model(model_name):
    """Check if the model is an Outcome Reward Model (ORM)."""
    return 'orm' in model_name.lower()

###########################################################################
# ORM-specific functions
###########################################################################

def evaluate_with_orm(model, tokenizer, question, answer, device):
    """
    Evaluate an answer using an Outcome Reward Model (ORM).
    
    Args:
        model: The ORM model
        tokenizer: The tokenizer for the model
        question: The question text
        answer: The answer text
        device: The device to run inference on
        
    Returns:
        The score assigned by the ORM model
    """
    # Create a conversation with the question and answer
    conversation = []
    
    # Format the prompt based on the provided script
    conversation.append({"content": question + " " + answer, "role": "user"})
    conversation.append({"content": "+", "role": "assistant"})
    
    # Tokenize using the chat template
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)
    
    # Get token IDs for + and - (using the last token for each as in the worker function)
    plus_tag_id = tokenizer.encode('+')[-1]
    minus_tag_id = tokenizer.encode('-')[-1]
    candidate_tokens = [plus_tag_id, minus_tag_id]
    
    # Get model prediction
    with torch.no_grad():
        # Following the sample script: focus on the position that predicts +/-
        # Using -3 position as in the provided script
        logits = model(input_ids).logits[:, -3, candidate_tokens]
        
        # Get probability of "+" token (index 0 of candidate_tokens)
        scores = logits.softmax(dim=-1)[:, 0]  # 0 means prob of +
        score = scores[0].item()
    
    return score

###########################################################################
# PRM Evaluation pipeline
###########################################################################

def evaluate_model_with_reward_model(reward_model, reward_tokenizer, model_name, dataset_name, dataset_path, question_key,
                           answer_path, num_files, device, output_dir="reward_model_evaluation_results",
                           step_tag="ки", interval=5):
    """
    Evaluate a model on a specified dataset using a reward model (PRM or ORM) to find the best answer.
    
    Args:
        reward_model: Process or Outcome Reward Model
        reward_tokenizer: Tokenizer for the reward model
        model_name: Name of the model
        dataset_name: Name of dataset
        answer_path: Path to generated answers directory
        num_files: Number of files to evaluate
        device: CUDA device to use
        output_dir: Directory to save evaluation results
        step_tag: Step tag for Shepherd model
        interval: Interval for calculating results (e.g., 5 will calculate for 5, 10, 15, ..., num_files)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine if we're using ORM or PRM
    is_orm = is_orm_model(model_name)
    reward_model_type = "ORM" if is_orm else "PRM"
    
    print(f"Evaluating {model_name} on {dataset_name} using {reward_model_type}")
    
    # Load dataset and get ground truth answers
    dataset, ground_truth_answers = get_dataset_and_answers(dataset_name, dataset_path, question_key)
    
    # Load and preprocess answers
    all_answers = load_json_answers(answer_path, num_files=num_files)
    cleaned_answers, questions = clean_all_answers(all_answers)
    
    # Combine all answers for evaluation
    all_cleaned_answers = []
    for i in range(len(cleaned_answers[0])):
        question_answers = []
        for j in range(len(cleaned_answers)):
            question_answers.append(cleaned_answers[j][i])
        all_cleaned_answers.append(question_answers)
    
    # Evaluate each answer with reward model and store all scores
    all_answer_scores = []
    
    # Define tokens for Shepherd model
    good_token = '+'
    bad_token = '-'
    
    # Check if using Shepherd model
    is_shepherd = is_shepherd_model(model_name)
    
    start_time = time.time()
    # Process all answers once
    for i, q_answers in enumerate(tqdm(all_cleaned_answers, desc=f"Evaluating with {reward_model_type}")):
        question = questions[i] if i < len(questions) else ""
        
        # Evaluate all answers for this question
        answer_scores = []
        for answer in q_answers:
            if is_orm:
                # ORM evaluation - just evaluate the full answer
                score = evaluate_with_orm(
                    model=reward_model,
                    tokenizer=reward_tokenizer,
                    question=question,
                    answer=answer,
                    device=device
                )
            else:
                # PRM evaluation
                # Split answer into sentences
                sentences = split_into_sentences(answer)
                if not sentences:
                    answer_scores.append((answer, 0.0))
                    continue
                
                if is_shepherd:
                    # Format sentences with "Step n:" prefix and step_tag
                    formatted_sentences = []
                    for step_idx, sentence in enumerate(sentences):
                        # Split sentence into lines to preserve existing newlines
                        lines = sentence.split('\n')
                        formatted_lines = []
                        
                        # Process all lines except the last one
                        if len(lines) > 1:
                            formatted_lines.extend(lines[:-1])
                        
                        # Add step tag to the last line if there is one
                        if lines:
                            formatted_lines.append(f"{lines[-1]} {step_tag}")
                        
                        # Rejoin lines with newlines and add step prefix
                        if formatted_lines:
                            formatted_sentence = f"Step {step_idx+1}: " + '\n'.join(formatted_lines) + "\n"
                            formatted_sentences.append(formatted_sentence)
                    
                    # Join all formatted sentences
                    formatted_answer = "".join(formatted_sentences)
                    
                    # Shepherd model evaluation
                    input_for_prm = f"{question} {formatted_answer}"
                    
                    input_ids = torch.tensor([reward_tokenizer.encode(input_for_prm)]).to(device)
                    
                    with torch.no_grad():
                        logits = reward_model(input_ids).logits[:,:,reward_tokenizer.encode(f"{good_token} {bad_token}")[1:]]
                        scores = logits.softmax(dim=-1)[:,:,0]
                        step_scores = scores[input_ids == reward_tokenizer.encode(f"{step_tag}")[-1]]
                        
                    if len(step_scores) > 0:
                        score = calculate_aggregate_score(step_scores.tolist())
                    else:
                        score = 0.0
                else:
                    # Format data for PRM
                    data = {
                        "system": "Please reason step by step, and put your final answer within \\boxed{}.",
                        "query": question,
                        "response": sentences
                    }
                    
                    # Create messages for the model
                    messages = [
                        {"role": "system", "content": data['system']},
                        {"role": "user", "content": data['query']},
                        {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
                    ]
                    
                    # Apply chat template
                    conversation_str = reward_tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
                    
                    # Encode and run inference
                    input_ids = reward_tokenizer.encode(
                        conversation_str, 
                        return_tensors="pt", 
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = reward_model(input_ids=input_ids)
                    
                    # Calculate step rewards
                    step_sep_id = reward_tokenizer.encode("<extra_0>")[0]
                    token_masks = (input_ids == step_sep_id)
                    step_reward = make_step_rewards(outputs[0], token_masks)
                    
                    # Calculate aggregate score
                    if step_reward and len(step_reward) > 0 and len(step_reward[0]) > 0:
                        score = calculate_aggregate_score(step_reward[0])
                    else:
                        score = 0.0
            
            answer_scores.append((answer, score))
        
        # Store all answers and scores for this question
        all_answer_scores.append(answer_scores)
    total_time = time.time() - start_time
    print(f"Time taken to evaluate {num_files} files: {total_time:.2f} seconds")
    # Calculate results for different intervals
    intervals = []
    if interval > 0:
        intervals = list(range(interval, num_files + 1, interval))
        if num_files not in intervals:
            intervals.append(num_files)
    else:
        # If interval is 0, just use the full set
        intervals = [num_files]
    
    # Calculate results for each interval
    for curr_num_files in intervals:
        print(f"\nCalculating results for {curr_num_files} files...")
        
        # Select best answers based on the first curr_num_files scores
        best_answers = []
        best_scores = []
        
        for question_scores in all_answer_scores:
            # Get only the first curr_num_files scores
            limited_scores = question_scores[:curr_num_files]
            
            if limited_scores:
                best_answer, best_score = max(limited_scores, key=lambda x: x[1])
                best_answers.append(best_answer)
                best_scores.append(best_score)
            else:
                best_answers.append("")
                best_scores.append(0.0)
        
        # Extract final answers from best solutions
        extracted_answers = []
        
        for answer in tqdm(best_answers, desc="Extracting answers"):
            # Extract boxed answer or answer after "The answer is"
            extracted_answer = extract_answer_from_text(answer)
            
            # Normalize the answer
            extracted_answer = normalize_answer(extracted_answer)
            extracted_answers.append(extracted_answer)
        
        # Calculate accuracy
        accuracy = compare_with_ground_truth(extracted_answers, ground_truth_answers)
        
        # Generate output filename
        cleaned_model_name = model_name.split('model/checkpoints/')[-1].split('/')[0] if 'model/checkpoints/' in model_name else model_name.replace('/', '_')
        reward_type = "orm" if is_orm else "prm"
        output_filename = f"{cleaned_model_name}_{dataset_name}_{curr_num_files}answers_{reward_type}_results.json"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save results
        results = {
            'model_name': model_name,
            'dataset': dataset_name,
            'reward_model_type': reward_model_type,
            'num_files': curr_num_files,
            'accuracy': accuracy,
            'best_scores': best_scores,
            'best_answers': best_answers,
            'extracted_answers': extracted_answers,
            'ground_truth': ground_truth_answers,
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"{dataset_name.upper()} {reward_model_type} Evaluation Results with {curr_num_files} files:")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Results saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate language models on various math datasets using reward models")
    parser.add_argument("--reward_model_path", type=str, default="model/Qwen2.5-Math-PRM-7B",
                        help="Path to the reward model checkpoint (PRM or ORM)")
    parser.add_argument("--answer_path", type=str, required=True,
                        help="Path to the generated answers directory")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device to use (e.g., cuda:0, cuda:1)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results/reward_models",
                        help="Directory to save results")
    parser.add_argument("--num_files", type=int, default=5,
                        help="Number of files to evaluate")
    parser.add_argument("--step_tag", type=str, default="ки",
                        help="Step tag for Shepherd model (default: ки)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Interval for calculating results (default: 5, 0 means only calculate for num_files)")
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Path to the dataset to use (default: "")")
    parser.add_argument("--question_key", type=str, default="question",
                        help="Key to the question in the dataset (default: question)")
    args = parser.parse_args()
    
    # Determine dataset type from answer_path
    answer_path_lower = args.answer_path.lower()
    if "gsm8k" in answer_path_lower:
        dataset = "gsm8k"
    elif "aime24" in answer_path_lower:
        dataset = "aime24"
    elif "aime25" in answer_path_lower:
        dataset = "aime25"
    elif "amc23" in answer_path_lower:
        dataset = "amc23"
    elif "olympiad" in answer_path_lower:
        dataset = "olympiad"
    else:
        dataset = "math"  # default to math if no other dataset is detected
    
    # Check if using ORM or PRM
    is_orm = is_orm_model(args.reward_model_path)
    reward_model_type = "ORM" if is_orm else "PRM"
    
    # Load reward model and tokenizer
    print(f"Loading {reward_model_type} model from {args.reward_model_path}")
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    
    if is_shepherd_model(args.reward_model_path):
        reward_model = AutoModelForCausalLM.from_pretrained(
            args.reward_model_path,
            device_map=args.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
    elif is_orm_model(args.reward_model_path):
        reward_model = AutoModelForCausalLM.from_pretrained(
            args.reward_model_path, 
            device_map=args.device, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
    else:
        # For standard PRM models
        reward_model = AutoModel.from_pretrained(
            args.reward_model_path, 
            device_map=args.device, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
    
    # Run evaluation with reward model
    try:
        evaluate_model_with_reward_model(
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            model_name=args.reward_model_path,
            dataset_name=dataset,
            dataset_path=args.dataset_path,
            question_key=args.question_key,
            answer_path=args.answer_path,
            num_files=args.num_files,
            device=args.device,
            output_dir=args.output_dir,
            step_tag=args.step_tag,
            interval=args.interval
        )
    except Exception as e:
        print(f"Error during {reward_model_type} evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()