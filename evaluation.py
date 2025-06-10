import os
import json
import re
import torch
import argparse
from tqdm import tqdm
import sys
from collections import Counter
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Add path for the grader module
sys.path.append('prm800k/prm800k')
from grading import grader, math_normalize

###########################################################################
# Utility functions
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

def extract_answer_from_aggregator(generated_text):
    """Extract answer from aggregator output."""
    matches = re.findall(r'<answer>(.*?)</answer>', generated_text)
    if matches:
        return matches[-1]
    return ""

def compare_with_ground_truth(predicted_answers, ground_truth_answers, show_progress=True, progress_desc="Grading Answers"):
    """Compare predicted answers with ground truth."""
    correct_count = 0
    
    iterable = zip(predicted_answers, ground_truth_answers)
    if show_progress:
        iterable = tqdm(iterable, total=len(ground_truth_answers) if ground_truth_answers else 0, desc=progress_desc)
        
    for pred, gt in iterable:
        if grader.grade_answer(pred, gt):
            correct_count += 1
    
    if not ground_truth_answers or len(ground_truth_answers) == 0: # Avoid division by zero
        return 0.0
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
    return answer.split('user\n')[1].split('assistant\n')[0].strip()

def normalize_gsm8k_format(answer):
    """Normalize GSM8K answer format."""
    return answer.replace('The Answer is:', '####')

def batch_clean_generated_answers(one_answer):
    """Clean a batch of generated answers."""
    cleaned_answers = []
    questions = []
    for answer in one_answer:
        # Extract question based on format of simpleRL
        if '[Round 0]' in answer:
            if 'ASSISTANT:' in answer:
                question = answer.split('USER:')[1].split('ASSISTANT:')[0].strip()
            else:
                question = answer.split('USER:')[1].strip()
        else:
            question = extract_question(answer)
        questions.append(question)
        
        # Clean answer based on format
        if '[Round 0]' in answer:
            if 'ASSISTANT:' in answer:
                answer = answer.split('ASSISTANT:')[1].strip()
                # Look for boxed answer
                match = re.search(r'\\boxed\{(.*?)\}', answer)
                if match:
                    answer = match.group(1).strip()
            else:
                answer = 'None'
        else:
            answer = remove_system_prompt(answer)
            answer = normalize_gsm8k_format(answer)
        cleaned_answers.append(answer)
    return cleaned_answers, questions

def clean_all_answers(all_answers):
    """Clean all generated answers."""
    cleaned_answers = []
    for answer in all_answers:
        extracted_answers, questions = batch_clean_generated_answers(answer)
        cleaned_answers.append(extracted_answers)
    return cleaned_answers, questions

def concatenate_answers(cleaned_answers, questions, prompt, num_answers=5):
    """Concatenate answers with questions."""
    concatenated_answers = []
    for j in range(len(cleaned_answers[0])):
        answer_concat = 'Question: ' + questions[j] + '\n\n'
        answer_concat += "Below are the proposed answers:"
        for i in range(num_answers):
            answer_prefix = f"\n\nAnswer {i+1}:\n"
            answer_concat += answer_prefix + cleaned_answers[i][j]
        answer = prompt.format(answer_concat)
        concatenated_answers.append(answer)
    return concatenated_answers

def get_gsm8k_answers(gsm8k_dataset):
    """Extract answers from GSM8K dataset."""
    gsm8k_answers = [x.split('####')[1].strip() for x in gsm8k_dataset['answer']]
    return gsm8k_answers

def get_dataset_and_answers(dataset_name, dataset_path, answer_key, split='test'):
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
    elif dataset_name == 'arc':
        dataset = load_dataset("allenai/ai2_arc", 'ARC-Challenge', split=split)
        ground_truth_answers = dataset['answerKey']
        # Ensure ground truth answers are strings for consistency
        ground_truth_answers = [str(x) for x in ground_truth_answers]
    elif dataset_name == 'mmlu-pro':
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
        ground_truth_answers = dataset['answer']
        ground_truth_answers = [str(x) for x in ground_truth_answers] # Ensure string type
    elif dataset_name == 'truthfulqa':
        dataset = load_dataset("EleutherAI/truthful_qa_mc", split='validation')
        # Convert numeric labels to letters (0->A, 1->B, etc.)
        ground_truth_answers = [chr(65 + x) for x in dataset['label']]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: gsm8k, math, aime24, aime25, amc23, olympiad, arc, mmlu-pro, truthfulqa")
    
    return dataset, ground_truth_answers

###########################################################################
# Evaluation pipeline
###########################################################################

def evaluate_model(model, tokenizer, model_name, dataset_name, dataset_path, answer_key, answer_path, num_answers, device, 
                   output_dir="evaluation_results", remove_answer_percentage=0.0):
    """
    Evaluate a model on a specified dataset.
    
    Args:
        model_path: Path to the model checkpoint
        dataset_name: Name of dataset ('gsm8k', 'math', 'aime24', 'aime25', 'amc23', 'olympiad')
        answer_path: Path to generated answers directory  
        num_answers: Number of answers to concatenate
        device: CUDA device to use
        output_dir: Directory to save evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Evaluating {model_name} on {dataset_name}")
    
    # Concatenate answers
    prompt = '{}'
    
    # Load dataset and get ground truth answers
    dataset, ground_truth_answers = get_dataset_and_answers(dataset_name, dataset_path, answer_key)
    
    # Load and preprocess answers
    all_answers = load_json_answers(answer_path)
    cleaned_answers, questions = clean_all_answers(all_answers)
    
    if remove_answer_percentage > 0:
        # Remove answers dynamically for each answer set
        cleaned_answers_removed = []
        for answer_set in cleaned_answers:
            answer_set_removed = []
            for answer in answer_set:
                num_answers_to_remove = int(len(answer) * remove_answer_percentage)
                if num_answers_to_remove > 0:
                    answer = answer[:-num_answers_to_remove]
                answer_set_removed.append(answer)
            cleaned_answers_removed.append(answer_set_removed)
        concatenated_answers = concatenate_answers(cleaned_answers_removed, questions, prompt, num_answers)
    else:
        concatenated_answers = concatenate_answers(cleaned_answers, questions, prompt, num_answers)

    
    # Create combined dataset
    combined_data = Dataset.from_dict({
        'question': concatenated_answers,
        'answer': ground_truth_answers
    })
    
    
    model.eval()
    
    if 'nothinking' in model_name:
         # Define prompt template
        prompt_template = (
            "A conversation between User and Assistant. The user provide a question and some proposed answers. The Assistant answer the question based on the proposed answers. The answer is enclosed within <answer></answer> tag, i.e., <answer>answer here</answer>. User: {}. Assistant: <answer>"
        )
    else:
        # Define prompt template
        prompt_template = (
            "A conversation between User and Assistant. The user provide a question and some proposed answers. The Assistant first evaluate each answers individually,check whether each answer directly addresses the original question, assess the correctness of each answer based on logical reasoning, calculations, and accuracy relative to the question. After thorough evaluation, identify one correct answer. If the correct answer is not in the provided proposed answers, the Assistant will combine the correct answer with the proposed answers and provide the correct answer."
            "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, "
            "i.e., <think>reasoning process here</think> <answer>answer here</answer>. User: {}. Assistant: <think>"
        )
    system_prompt = "You are a helpful assistant. The user provide a question and some proposed answers. The Assistant first evaluate each answers individually,check whether each answer directly addresses the original question, assess the correctness of each answer based on logical reasoning, calculations, and accuracy relative to the question. After thorough evaluation, identify one correct answer based on majority consensus. The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think>reasoning process here</think> <answer>answer here</answer>."

    start_time = time.time()
    # Generate answers
    sequential_answers = []
    for i in tqdm(range(len(combined_data['question'])), desc=f"Evaluating {dataset_name}"):
        if 'Instruct' in model_name:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_data['question'][i]}
            ]
            test_answer = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([test_answer], max_length=(num_answers+2)*1024, return_tensors="pt").to(device)
        else:
            test_answer = prompt_template.format(combined_data['question'][i])
            inputs = tokenizer(test_answer, max_length=(num_answers+2)*1024, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                top_k=40,
                temperature=0.1,
                num_return_sequences=1,
            )
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            sequential_answers.append(generated_text)
    total_time = time.time() - start_time
    print(f"Time taken to generate answers: {total_time:.2f} seconds")
    # Extract and evaluate answers
    extracted_answers = []
    for answer in tqdm(sequential_answers, desc="Extracting answers"):
        extracted_answer = extract_answer_from_aggregator(answer)
        extracted_answers.append(extracted_answer)
    
    # Calculate accuracy
    accuracy = compare_with_ground_truth(extracted_answers, ground_truth_answers)
    
    results = {
        'model_name': model_name,
        'dataset': dataset_name,
        'num_answers': num_answers,
        'accuracy': accuracy,
        'sequential_answers': sequential_answers,
        'extracted_answers': extracted_answers,
        'ground_truth': ground_truth_answers,
        'category_accuracies': {} # Initialize for mmlu-pro
    }

    if dataset_name == "mmlu-pro":
        print("\nCalculating category-specific accuracies for MMLU-Pro...")
        unique_categories = sorted(list(set(dataset['category'])))
        category_accuracies_dict = {}
        for cat in tqdm(unique_categories, desc="Evaluating categories"):
            category_indices = [i for i, item_category in enumerate(dataset['category']) if item_category == cat]
            
            cat_extracted_answers = [extracted_answers[i] for i in category_indices]
            cat_ground_truth_answers = [ground_truth_answers[i] for i in category_indices]
            
            if len(cat_ground_truth_answers) > 0:
                cat_accuracy = compare_with_ground_truth(cat_extracted_answers, cat_ground_truth_answers, show_progress=False)
                category_accuracies_dict[cat] = cat_accuracy
                print(f"Accuracy for category '{cat}': {cat_accuracy*100:.2f}% (samples: {len(cat_ground_truth_answers)})")
            else:
                category_accuracies_dict[cat] = 0.0
                print(f"No samples for category '{cat}'.")
        results['category_accuracies'] = category_accuracies_dict
        
    # Generate output filename
    cleaned_model_name = model_name.split('model/checkpoints/')[-1].split('/')[0]
    output_filename = f"{cleaned_model_name}_{dataset_name}_concat{num_answers}_results.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{dataset_name.upper()} Evaluation Results:")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Results saved to: {output_path}\n")
    
    return accuracy, output_path

def main():
    parser = argparse.ArgumentParser(description="Evaluate language models on various math datasets")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--answer_path", type=str, required=True,
                        help="Path to the generated answers directory")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device to use (e.g., cuda:0, cuda:1)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save results")
    parser.add_argument("--num_answers", type=int, default=5,
                        help="max number of answers to concatenate")
    parser.add_argument("--start_num_answers", type=int, default=5,
                        help="start number of answers to concatenate")
    parser.add_argument("--remove_answer_percentage", type=float, default=0.0,
                        help="percentage of answers to remove")
    parser.add_argument("--step_num_answers", type=int, default=5,
                        help="step number of answers to concatenate")
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Path to the dataset to use (default: "")")
    parser.add_argument("--answer_key", type=str, default="answer",
                        help="Key to the answer in the dataset (default: answer)")
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
    elif "arc" in answer_path_lower:
        dataset = "arc"
    elif "mmlu-pro" in answer_path_lower:
        dataset = "mmlu-pro"
    elif "truthfulqa" in answer_path_lower:
        dataset = "truthfulqa"
    else:
        dataset = "math"  # default to math if no other dataset is detected
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    
    # List of num_answers to evaluate [5, 10, 15, 20]
    num_answers_list = list(range(args.start_num_answers, args.num_answers+1, args.step_num_answers))
    print(num_answers_list)
    # Evaluate for each num_answers value
    for num_answers in num_answers_list:
        try:
            evaluate_model(
                model=model,
                tokenizer=tokenizer,
                model_name=args.model_path,
                dataset_name=dataset,
                dataset_path=args.dataset_path,
                answer_key=args.answer_key,
                answer_path=args.answer_path,
                num_answers=num_answers,
                device=args.device,
                output_dir=args.output_dir,
                remove_answer_percentage=args.remove_answer_percentage
            )
        except Exception as e:
            print(f"Error evaluating with num_answers={num_answers}: {str(e)}")
            continue

if __name__ == "__main__":
    main()