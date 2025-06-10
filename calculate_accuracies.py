import os
import json
import argparse
import re
from tqdm import tqdm
from collections import Counter
import sys
from datasets import load_dataset

# Add path for the grader module
sys.path.append('prm800k/prm800k')
from grading import grader, math_normalize

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
    #SimpleRL Results extractions
    elif 'ASSISTANT:' in text:
        if '[ \\boxed{' in text.split('ASSISTANT:')[1]:
            answer = text.split('ASSISTANT:')[1].split('[ \\boxed{')[1]
            #Have to split by } because the answer from SimpleRL is really bad...
            # repeating some random text after the answer
            answer = answer.split('\\]')[0]
            answer = r'\\boxed{' + answer
            return answer.strip()
        elif 'boxed{' in text.split('ASSISTANT:')[1]:
            answer = text.split('ASSISTANT:')[1].split('boxed{')[1]
            answer = r'\\boxed{' + answer
            return answer.strip()
        else:
            return None
    else:
        return None

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

def extract_answers_from_nested_list(nested_answer_texts):
    """Extract answers from nested list of generated texts."""
    extracted = []
    for answer_list in nested_answer_texts:
        extracted_file = []
        for ans in answer_list:
            if isinstance(ans, str):
                answer = clean_generated_text(ans)
                normalized_answer = normalize_answer(answer)
                extracted_file.append(normalized_answer)
            else:
                extracted_file.append(None)
        extracted.append(extracted_file)
    return extracted

def transpose_nested_list(nested_list):
    """Transpose a nested list."""
    return [list(t) for t in zip(*nested_list)]

def majority_vote(extracted_answers):
    """Select most frequent answer for each question."""
    majority_answers = []
    for question_answers in extracted_answers:
        valid_answers = [answer for answer in question_answers if answer and answer.strip()]
        if valid_answers:
            counter = Counter(valid_answers)
            majority_answer = counter.most_common(1)[0][0]
        else:
            majority_answer = ""
        majority_answers.append(majority_answer)
    return majority_answers

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

def get_dataset_and_answers(dataset_name, dataset_path, answer_key, split='test'):
    """Load dataset and extract answers based on dataset type."""
    if dataset_name == 'gsm8k':
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        ground_truth_answers = [x.split('####')[1].strip() for x in dataset['answer']]
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
        ground_truth_answers = [str(x) for x in ground_truth_answers]
    elif dataset_name == 'mmlu-pro':
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
        ground_truth_answers = dataset['answer']
        ground_truth_answers = [str(x) for x in ground_truth_answers]
    elif dataset_name == 'truthfulqa':
        dataset = load_dataset("EleutherAI/truthful_qa_mc", split='validation')
        # Convert numeric labels to letters (0->A, 1->B, etc.)
        ground_truth_answers = [chr(65 + x) for x in dataset['label']]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: gsm8k, math, aime24, aime25, amc23, olympiad, arc, mmlu-pro, truthfulqa")
    
    return dataset, ground_truth_answers

def determine_dataset_from_path(answer_path):
    """Determine dataset type from the answer path."""
    answer_path_lower = answer_path.lower()
    if "gsm8k" in answer_path_lower:
        return "gsm8k"
    elif "aime24" in answer_path_lower:
        return "aime24"
    elif "aime25" in answer_path_lower:
        return "aime25"
    elif "amc23" in answer_path_lower:
        return "amc23"
    elif "olympiad" in answer_path_lower:
        return "olympiad"
    elif "arc" in answer_path_lower:
        return "arc"
    elif "mmlu-pro" in answer_path_lower:
        return "mmlu-pro"
    elif "truthfulqa" in answer_path_lower:
        return "truthfulqa"
    elif "math" in answer_path_lower:
        return "math"
    else:
        raise ValueError(f"Could not determine dataset type from path: {answer_path}. Ensure path contains dataset name like 'gsm8k', 'math', 'aime24', 'aime25', 'amc23', 'olympiad', 'arc', 'mmlu-pro', or 'truthfulqa'.")

def max_accuracy(extracted, ground_truth_answers):
    """
    Check the answers accuracy with the ground truth. If at least one answer is correct, then the question is considered correct.
    """
    correct_count = 0
    for pred, gt in tqdm(zip(extracted, ground_truth_answers),
                         total=len(ground_truth_answers),
                         desc="Grading Answers"):
        for answer in pred:
            if grader.grade_answer(answer, gt):
                correct_count += 1
                break
    return correct_count / len(ground_truth_answers)

def calculate_accuracies(answer_path, number_of_files, dataset_path, answer_key):
    """Calculate and display various accuracy metrics."""
    # Determine dataset from path
    dataset_name = determine_dataset_from_path(answer_path)
    
    # Load dataset and ground truth
    _, ground_truth_answers = get_dataset_and_answers(dataset_name, dataset_path, answer_key)
    
    # Load all answers
    all_answers = load_json_answers(answer_path, number_of_files)
    extracted = extract_answers_from_nested_list(all_answers)
    extracted = transpose_nested_list(extracted)
    
    # Calculate single file accuracy (first file)
    single_file_answers = [answers[0] if answers else None for answers in extracted]
    single_file_accuracy = compare_with_ground_truth(single_file_answers, ground_truth_answers)
    
    # Calculate 5 files accuracy using max_accuracy
    five_files_answers = [answers[:] for answers in extracted]
    five_files_accuracy = max_accuracy(five_files_answers, ground_truth_answers)
    
    # Calculate majority vote accuracy
    majority_ans = majority_vote(extracted)
    majority_accuracy = compare_with_ground_truth(majority_ans, ground_truth_answers)
    
    # Print results
    print(f"\nAccuracy Results for {dataset_name.upper()}:")
    print(f"Single File Accuracy: {single_file_accuracy*100:.2f}")
    print(f"{number_of_files} Majority Vote Accuracy: {majority_accuracy*100:.2f}")
    print(f"{number_of_files} Files Max Accuracy: {five_files_accuracy*100:.2f}")
    return single_file_accuracy, majority_accuracy, five_files_accuracy

def main():
    parser = argparse.ArgumentParser(description="Calculate and display various accuracy metrics")
    parser.add_argument("--answer_path", type=str, required=True,
                        help="Path to the generated answers directory")
    parser.add_argument("--num_files", type=int, required=True,
                        help="Number of files to calculate accuracy for")
    parser.add_argument("--set_number", type=int, required=False, default=0,
                        help="Number of files to calculate accuracy for")
    parser.add_argument("--dataset_path", type=str, required=False, default='',
                        help="Path to the dataset")
    parser.add_argument("--answer_key", type=str, required=False, default='',
                        help="Key to the answer in the dataset")
    args = parser.parse_args()
    
    number_list = list(range(5, args.num_files+1, 5))
    if args.set_number > 0:
        number_list = [args.set_number]
    single_file_accuracy_list = []
    majority_accuracy_list = []
    five_files_accuracy_list = []
    for number in number_list:
        single_file_accuracy, majority_accuracy, five_files_accuracy = calculate_accuracies(args.answer_path, number, args.dataset_path, args.answer_key)
        single_file_accuracy_list.append(single_file_accuracy)
        majority_accuracy_list.append(majority_accuracy)
        five_files_accuracy_list.append(five_files_accuracy)
    print('--------------------------------')
    print(f"Accuracy Results for {args.answer_path.split('/')[-1]}:")
    print(f"Single File Accuracy: {single_file_accuracy_list[0]*100:.2f}")
    for i in range(len(number_list)):
        print(f"Number of files: {number_list[i]}")
        print(f"{number_list[i]} Majority Vote Accuracy: {majority_accuracy_list[i]*100:.2f}")
        print(f"{number_list[i]} Files Max Accuracy: {five_files_accuracy_list[i]*100:.2f}")
        print("\n")
    print('--------------------------------')
if __name__ == "__main__":
    main() 