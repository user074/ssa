#####################################################################
####### Helper functions #######
#####################################################################
import os
import json
import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# Append path for the grader module (your custom grading function)
sys.path.append('prm800k/prm800k')
from grading import grader, math_normalize


def load_json_answers(directory, num_files=None):
    """Load generated answers from all JSON files in the given directory."""
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
    if 'assistant\n' not in answer:
        return answer.strip()
    return answer.split('assistant\n')[1].strip()

def extract_question(answer):
    return answer.split('user\n')[1].split('assistant\n')[0].strip()

def normalize_gsm8k_format(answer):
    #replace the answer from 'The Answer is:' to '####'
    return answer.replace('The Answer is:', '####')

def batch_clean_generated_answers(one_answer):
    cleaned_answers = []
    questions = []
    for answer in one_answer:
        question = extract_question(answer)
        questions.append(question)
        answer = remove_system_prompt(answer)
        answer = normalize_gsm8k_format(answer)
        cleaned_answers.append(answer)
    return cleaned_answers, questions

def clean_all_answers(all_answers):
    cleaned_answers = []
    for answer in all_answers:
        extracted_answers, questions = batch_clean_generated_answers(answer)
        cleaned_answers.append(extracted_answers)
    return cleaned_answers, questions

def concatenate_answers(cleaned_answers, questions, prompt, num_answers=5):
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

def concatenate_answers_random(cleaned_answers, questions, prompt, min_answers=3, max_answers=5):
    import random
    concatenated_answers = []
    for j in range(len(cleaned_answers[0])):
        answer_concat = 'Question: ' + questions[j] + '\n\n'
        answer_concat += "Below are the proposed answers:"
        num_answers = random.randint(min_answers, max_answers)
        for i in range(num_answers):
            answer_prefix = f"\n\nAnswer {i+1}:\n"
            answer_concat += answer_prefix + cleaned_answers[i][j]
        answer = prompt.format(answer_concat)
        concatenated_answers.append(answer)
    return concatenated_answers


###########################################################################
# Helper functions for loading and extracting generated answers
###########################################################################

def normalize_answer(ans):
    """
    Normalize a generated answer string by:
      1. Returning None if the input is None.
      2. Stripping leading/trailing whitespace.
      3. Removing surrounding '$' characters.
      4. Removing a \boxed{...} wrapper if present.
    """
    if ans is None:
        return None

    # Ensure we are dealing with a string.
    if not isinstance(ans, str):
        ans = str(ans)
    ans = math_normalize.normalize_answer(ans)
    # Strip leading/trailing whitespace.
    ans = ans.strip()
    
    # Remove surrounding dollar signs, if present.
    if ans.startswith('$') and ans.endswith('$'):
        ans = ans[1:-1].strip()
    
    # Remove \boxed{...} wrapper if it exists.
    # This regex looks for a literal "\boxed{" at the start and "}" at the end.
    match = re.fullmatch(r'\\boxed\{(.*)\}', ans)
    if match:
        ans = match.group(1).strip()
    
    return ans

def clean_generated_text(text):
    """
    Remove any spurious text from the generated output.
    For example, remove occurrences of "The Answer is: user"
    so that only "The Answer is:" remains.
    """
    #might be none if the answer is not in the generated text
    if 'assistant\n' in text:
        if 'is:' in text.split('assistant\n')[1]:
            answer = text.split('assistant\n')[1].split("is:")[-1]
            if '\\' in answer:
                return answer.strip()
            words = answer.split()
            if len(words)>1:
                try:
                    float(words[0])
                    return words[0]
                except:
                    return answer.strip()
                
                
            #need to check if the answer contains units or not
            #get the first item seperated by space after the last "The Answer is:"
            # answer = text.split('assistant\n')[1].split("The Answer is: ")[-1].split(" ")[0]
            return answer.strip()
        elif 'boxed{' in text.split('assistant\n')[1]:
            answer = text.split('assistant\n')[1].split('boxed{')[1]
            answer = r'\(\boxed{' + answer
            return answer.strip()
        else:
            return None
    else:
        return None

def extract_answers_from_nested_list(nested_answer_texts):
    """
    Given a nested list (list of lists) of generated answer texts,
    extract the answer from each string.
    
    Returns a nested list of the extracted answers.
    """
    extracted = []
    for answer_list in nested_answer_texts:
        extracted_file = []
        for ans in answer_list:
            if isinstance(ans, str):
                answer = clean_generated_text(ans)
                normalized_answer = normalize_answer(answer)
                extracted_file.append(normalized_answer)
            else:
                # If ans is not a string, skip it or return None.
                extracted_file.append(None)
        extracted.append(extracted_file)
    return extracted


def transpose_nested_list(nested_list):
    """
    Transpose a nested list (list of lists).
    """
    return [list(t) for t in zip(*nested_list)]

#####################################################################
####### Helper functions for filtering invalid answers #######
#####################################################################

def filter_none_values(math_answers_from_json, allowed_none_counts=None):
    """
    Filter out answers with too many None values.
    """
    if allowed_none_counts is None:
        allowed_none_counts = [2, 3, 4, 5]
        
    extracted = extract_answers_from_nested_list(math_answers_from_json)
    extracted = transpose_nested_list(extracted)
    filtered_answers_index = []
    
    for i, item in enumerate(extracted):
        none_count = sum(x is None for x in item)
        if none_count not in allowed_none_counts:
            filtered_answers_index.append(i)
    
    filtered_math_answers_from_json = []
    for i in range(len(math_answers_from_json)):
        filtered_one_answer = []
        for j in filtered_answers_index:
            filtered_one_answer.append(math_answers_from_json[i][j])
        filtered_math_answers_from_json.append(filtered_one_answer)
    
    print(f"Number of filtered answers: {len(extracted)-len(filtered_answers_index)}")
    print(f"Number of answers remain: {len(filtered_answers_index)}")
    
    return filtered_answers_index, filtered_math_answers_from_json

def get_filtered_answers(filtered_answers_index, answers):
    filtered_answers = []
    for i in filtered_answers_index:
        filtered_answers.append(answers[i])
    return filtered_answers


def get_gsm8k_answers(gsm8K_train_dataset):
    gsm8K_answers = [x.split('####')[1].strip() for x in gsm8K_train_dataset['answer']]
    return gsm8K_answers
    

def remove_long_answers(answers, tokenizer, max_length=4000):
    token_lengths = [len(tokenizer.encode(answer)) for answer in answers]
    long_sequence_indices = [i for i, length in enumerate(token_lengths) if length <= max_length]
    cleaned_answers = [answers[i] for i in range(len(answers)) if i in long_sequence_indices]
    return cleaned_answers, long_sequence_indices, token_lengths

def plot_token_distribution(token_lengths, output_path=None):
    """
    Plot the distribution of token lengths.
    """
    mean_length = np.mean(token_lengths)
    median_length = np.median(token_lengths)
    max_length = np.max(token_lengths)
    min_length = np.min(token_lengths)
    std_length = np.std(token_lengths)

    print(f"Token length statistics:")
    print(f"Mean: {mean_length:.2f}")
    print(f"Median: {median_length:.2f}")
    print(f"Max: {max_length}")
    print(f"Min: {min_length}")
    print(f"Standard deviation: {std_length:.2f}")

    # Create histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=token_lengths, bins=50)

    # Customize plot
    plt.title('Distribution of Token Lengths in Answers')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Count')

    # Add vertical lines for statistics
    plt.axvline(mean_length, color='r', linestyle='--', label=f'Mean: {mean_length:.1f}')
    plt.axvline(median_length, color='g', linestyle='--', label=f'Median: {median_length:.1f}')

    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(os.path.join(output_path, 'token_length_histogram.png'))
    else:
        plt.show()

    # Create box plot
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=token_lengths)
    plt.title('Box Plot of Token Lengths')
    plt.xlabel('Number of Tokens')
    
    if output_path:
        plt.savefig(os.path.join(output_path, 'token_length_boxplot.png'))
    else:
        plt.show()
        

def process_datasets(args):
    """
    Main function to process the datasets.
    """
    # Set default prompt template if not provided
    prompt_template = args.prompt_template if args.prompt_template else '{}'
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    
    # Load answers
    gsm8k_answers_from_json = load_json_answers(args.gsm8k_path)
    math_answers_from_json = load_json_answers(args.math_path)
    
    # Filter answers
    gsm8k_filtered_answers_index, gsm8k_filtered_answers_from_json = filter_none_values(
        gsm8k_answers_from_json, 
        args.allowed_none_counts
    )
    
    math_filtered_answers_index, math_filtered_answers_from_json = filter_none_values(
        math_answers_from_json, 
        args.allowed_none_counts
    )
    
    # Clean answers
    gsm8k_cleaned_answers, gsm8k_questions = clean_all_answers(gsm8k_filtered_answers_from_json)
    math_cleaned_answers, math_questions = clean_all_answers(math_filtered_answers_from_json)
    
    # Create concatenated answers
    if args.random_answers:
        gsm8k_concatenated_answers = concatenate_answers_random(
            gsm8k_cleaned_answers, 
            gsm8k_questions, 
            prompt_template, 
            args.min_answers, 
            args.max_answers
        )
        
        math_concatenated_answers = concatenate_answers_random(
            math_cleaned_answers, 
            math_questions, 
            prompt_template, 
            args.min_answers, 
            args.max_answers
        )
    else:
        gsm8k_concatenated_answers = concatenate_answers(
            gsm8k_cleaned_answers, 
            gsm8k_questions, 
            prompt_template, 
            args.num_answers
        )
        
        math_concatenated_answers = concatenate_answers(
            math_cleaned_answers, 
            math_questions, 
            prompt_template, 
            args.num_answers
        )
    
    # Load ground truth datasets
    gsm8K_train_dataset = load_dataset(args.gsm8k_dataset, "main", split='train')
    math_train_dataset = load_dataset(args.math_dataset, split='train')
    
    # Get ground truth answers
    gsm8K_answers = get_gsm8k_answers(gsm8K_train_dataset)
    math_answers = math_train_dataset['answer']
    
    # Filter ground truth answers
    gsm8k_filtered_answers = get_filtered_answers(gsm8k_filtered_answers_index, gsm8K_answers)
    math_filtered_answers = get_filtered_answers(math_filtered_answers_index, math_answers)
    
    # Combine answers
    combined_generated_answers = gsm8k_concatenated_answers + math_concatenated_answers
    combined_ground_truth_answers = gsm8k_filtered_answers + math_filtered_answers
    
    # Remove long answers
    cleaned_combined_generated_answers, long_sequence_indices_combined, token_lengths_combined = remove_long_answers(
        combined_generated_answers, 
        tokenizer, 
        args.max_length
    )
    
    cleaned_combined_ground_truth_answers = get_filtered_answers(
        long_sequence_indices_combined, 
        combined_ground_truth_answers
    )
    print('--------------------------------')
    print(f"Final number of examples: {len(cleaned_combined_generated_answers)}")
    print('--------------------------------')
    #calculate the percentage of remaining answers
    percentage_remaining = len(cleaned_combined_generated_answers) / len(combined_generated_answers)
    print(f"Percentage of remaining answers after removing long answers: {percentage_remaining:.2%}")
    print('--------------------------------')
    
    # Create and save dataset
    train_new_data = Dataset.from_dict({
        'question': cleaned_combined_generated_answers,
        'answer': cleaned_combined_ground_truth_answers
    })
    
    # Save or push dataset
    if args.output_dataset:
        if args.push_to_hub:
            print(f"Pushing dataset to Hugging Face Hub: {args.output_dataset}")
            train_new_data.push_to_hub(args.output_dataset)
        else:
            print(f"Saving dataset to local directory: {args.output_dataset}")
            train_new_data.save_to_disk(args.output_dataset)
    
    # Plot token distribution
    if args.plot_distribution:
        filtered_token_lengths = [token_lengths_combined[i] for i in range(len(token_lengths_combined)) 
                                  if i in long_sequence_indices_combined]
        plot_token_distribution(filtered_token_lengths, args.output_path)
    
    return train_new_data

def main():
    parser = argparse.ArgumentParser(description='Process math answer datasets')
    
    parser.add_argument('--gsm8k_path', type=str, required=True,
                        help='Path to GSM8K answers')
    parser.add_argument('--math_path', type=str, required=True,
                        help='Path to MATH answers')
    parser.add_argument('--model_path', type=str, default='model/Qwen2.5-3B',
                        help='Path to model for tokenization')
    parser.add_argument('--output_dataset', type=str, default=None,
                        help='Output path for the processed dataset')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for plots and stats')
    parser.add_argument('--prompt_template', type=str, default='{}',
                        help='Prompt template for formatting answers')
    parser.add_argument('--num_answers', type=int, default=5,
                        help='Number of answers to include')
    parser.add_argument('--random_answers', action='store_true',
                        help='Use random number of answers')
    parser.add_argument('--min_answers', type=int, default=3,
                        help='Minimum number of answers when using random')
    parser.add_argument('--max_answers', type=int, default=5,
                        help='Maximum number of answers when using random')
    parser.add_argument('--max_length', type=int, default=4000,
                        help='Maximum token length for filtering')
    parser.add_argument('--plot_distribution', action='store_true',
                        help='Plot token length distribution')
    parser.add_argument('--push_to_hub', action='store_true',
                        help='Push dataset to Hugging Face Hub')
    parser.add_argument('--allowed_none_counts', type=int, nargs='+', 
                        default=[2, 3, 4, 5],
                        help='Allowed counts of None values in answers')
    parser.add_argument('--gsm8k_dataset', type=str, default='openai/gsm8k',
                        help='GSM8K dataset name')
    parser.add_argument('--math_dataset', type=str, default='user074/prm800k_MATH_splits',
                        help='MATH dataset name')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_path and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    process_datasets(args)

if __name__ == "__main__":
    main()