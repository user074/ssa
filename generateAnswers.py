import os
import json
import torch
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator


def main():
    ############################################################################
    # 0) Parse Arguments
    ############################################################################
    parser = argparse.ArgumentParser(description='Generate answers using a language model')
    parser.add_argument('--model_name', type=str, default="model/Qwen2.5-7B-Instruct",
                      help='Path to the model to use (default: model/Qwen2.5-7B-Instruct)')
    parser.add_argument('--dataset_name', type=str, default="gsm8k",
                      help='Dataset to use (options: gsm8k, MATH)')
    parser.add_argument('--split', type=str, default="test",
                      help='Dataset split to use (default: test)')
    parser.add_argument('--total_num_samples', type=int, default=8,
                      help='Number of samples to run (default: 8)')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                      help='Maximum number of tokens to generate (default: 1024)')
    parser.add_argument('--dataset_path', type=str, default="",
                      help='Path to the dataset to use (default: "")')
    parser.add_argument('--question_key', type=str, default="question",
                      help='Key to the question in the dataset (default: question)')
    args = parser.parse_args()

    # Extract the last part of model and dataset names
    model_name_short = args.model_name.split('/')[-1]
    dataset_name_short = args.dataset_name

    ############################################################################
    # 1) Initialize Accelerator
    ############################################################################
    accelerator = Accelerator()
    rank = accelerator.process_index       # which GPU/process is this?
    num_procs = accelerator.num_processes  # total GPUs/processes
    device = accelerator.device            # the device for this rank
    
    accelerator.print(f"[Rank {rank}] Starting script on device={device}.")

    ############################################################################
    # 2) Load Model & Tokenizer (CPU -> GPU)
    ############################################################################
    accelerator.print(f"[Rank {rank}] Loading tokenizer/model {args.model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model on CPU first (to avoid memory spikes on a single GPU)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16  
    )
    accelerator.print(f"[Rank {rank}] Moving model to {device} ...")
    model.to(device)
    model.eval()

    ############################################################################
    # 3) Load Dataset (No Splitting)
    ############################################################################
    accelerator.print(f"[Rank {rank}] Loading dataset {args.dataset_name} with split {args.split}.")
    
    # Dataset specific loading and formatting
    if args.dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split=args.split)
        question_key = "question"
    elif args.dataset_name == "MATH":
        dataset = load_dataset("user074/prm800k_MATH_splits", split=args.split)
        question_key = "problem"
    elif args.dataset_name == "aime24": 
        dataset = load_dataset("Maxwell-Jia/AIME_2024", split='train')
        question_key = "Problem"
    elif args.dataset_name == "aime25":
        dataset = load_dataset("yentinglin/aime_2025", split='train')
        question_key = "problem"
    elif args.dataset_name == "amc23":
        dataset = load_dataset("zwhe99/amc23", split='test')
        question_key = "question"
    elif args.dataset_name == "olympiad":
        dataset = load_dataset("Hothan/OlympiadBench", 'OE_TO_maths_en_COMP', split='train')
        question_key = "question"
    elif args.dataset_name == "mmlu-pro":
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split='test')
        choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
        
        def format_mmlu_pro_question(example):
            # Format the options
            options_text = ""
            for i, option in enumerate(example["options"]):
                if option != "N/A":
                    options_text += f"{choices[i]}. {option}\n"
            
            # Combine question and options
            formatted_question = f"{example['question']}\n\nOptions:\n{options_text}Please think step by step and answer concisely. Then select the correct answer choices (a single letter) at the end of response."
            
            return {"formatted_question": formatted_question}
        
        # Apply the formatting
        dataset = dataset.map(format_mmlu_pro_question)
        question_key = "formatted_question"
        
        accelerator.print(f"[Rank {rank}] Transformed MMLU-Pro dataset with {len(dataset)} samples.")
        
    elif args.dataset_name == "arc":
        dataset = load_dataset("allenai/ai2_arc", 'ARC-Challenge', split='test')
        
        def format_ai2_arc_question(example):
            # Format the options
            options_text = ""
            for label, text in zip(example["choices"]["label"], example["choices"]["text"]):
                options_text += f"{label}. {text}\n"
            
            # Combine question and options
            formatted_question = f"{example['question']}\n\nOptions:\n{options_text}Please think step by step and answer concisely. Then select the correct answer choices (a single letter) at the end of response."
            
            return {"formatted_question": formatted_question}
        
        # Apply the formatting
        dataset = dataset.map(format_ai2_arc_question)
        question_key = "formatted_question"
        
        accelerator.print(f"[Rank {rank}] Transformed AI2 ARC dataset with {len(dataset)} samples.")
        
    elif args.dataset_name == "truthfulqa":
        dataset = load_dataset("EleutherAI/truthful_qa_mc", split='validation')
        
        def format_truthfulqa_question(example):
            # Format the options
            options_text = ""
            for i, choice in enumerate(example["choices"]):
                options_text += f"{chr(65 + i)}. {choice}\n"
            
            # Combine question and options
            formatted_question = f"{example['question']}\n\nOptions:\n{options_text}Please think step by step and answer concisely. Then select the correct answer choices (a single letter) at the end of response."
            
            return {"formatted_question": formatted_question}
        
        # Apply the formatting
        dataset = dataset.map(format_truthfulqa_question)
        question_key = "formatted_question"
        
        accelerator.print(f"[Rank {rank}] Transformed TruthfulQA dataset with {len(dataset)} samples.")
        
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}. Please choose from: gsm8k, MATH, aime24, aime25, amc23, math24, mmlu-pro, arc, truthfulqa")

    # Simple formatting
    def prompt_format_func(examples):
        messages_list = [
            [
                {
                    "role": "system",
                    "content": "Cutting Knowledge Date: December 2023\nToday Date: 23 July 2024\n\nYou are a helpful assistant. Put the correct answer at the end of the message as 'The Answer is:'"
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
            for question in examples[question_key]
        ]
        
        return {
            "input_text": [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                for messages in messages_list
            ]
        }
    def prompt_format_func_simpleRL(examples):
        prompt_template = (
            "[Round 0] USER:\n{}\nPlease reason step by step, and put your final answer within \\boxed{{}}. ASSISTANT:\n"
        )
        #None chat message list
        messages_list = [ prompt_template.format(question) for question in examples[question_key] ]
        return {
            "input_text": messages_list
        }
    def prompt_format_func_Qwen_MATH(examples):
        messages_list = [
            [
                {
                    "role": "system",
                    "content": "Please reason step by step, and put your final answer within \\boxed{}."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
            for question in examples[question_key]
        ]
        
        return {
            "input_text": [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                for messages in messages_list
            ]
        }
    if 'SimpleRL' in args.model_name:
        dataset = dataset.map(prompt_format_func_simpleRL, batched=True)
    elif 'Qwen2.5-Math' in args.model_name:
        dataset = dataset.map(prompt_format_func_Qwen_MATH, batched=True)
    else:
        dataset = dataset.map(prompt_format_func, batched=True)
    input_texts = dataset["input_text"]

    accelerator.print(f"[Rank {rank}] total samples = {len(input_texts)}")
    start_time = time.time()
    ############################################################################
    # 4) Inference Loop
    ############################################################################

    num_passes = int(args.total_num_samples/num_procs)
    print(f"[Rank {rank}] num_passes = {num_passes}. Total samples = {args.total_num_samples}. num_procs = {num_procs}")
    batch_size = 1

    # Create output directory
    out_dir = os.path.join("answers", f"{model_name_short}_{dataset_name_short}_{args.total_num_samples}")
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    for pass_idx in range(num_passes):
        # Prepare an output list for this pass
        all_answers = []
        
        # Local progress bar (each GPU sees full data)
        pbar = tqdm(range(0, len(input_texts), batch_size), 
                    desc=f"[Rank {rank}] pass {pass_idx}", 
                    disable=not accelerator.is_main_process)
        
        for i in pbar:
            batch = input_texts[i : i + batch_size]
            batch_tok = tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **batch_tok,
                    max_new_tokens=args.max_new_tokens,      # Adjust to desired generation length
                    do_sample=True,
                    temperature=0.5,
                    top_k=40
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_answers.extend(decoded)

        # Save each rank's entire dataset answers
        out_file = f"answers_pass{pass_idx}_rank{rank}.json"
        out_path = os.path.join(out_dir, out_file)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_answers, f, ensure_ascii=False, indent=2)

        accelerator.print(f"[Rank {rank}] Pass={pass_idx} done. {len(all_answers)} answers -> {out_path}")

    total_time = time.time() - start_time
    accelerator.print(f"[Rank {rank}] Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
# accelerate launch --num_processes=4 generateAnswers.py --model_name "model/Qwen2.5-Math-7B-Instruct" --dataset_name "gsm8k" --split "test" --total_num_samples 8
# accelerate launch --num_processes=4 generateAnswers.py --model_name "model/Qwen2.5-Math-7B-Instruct" --dataset_name "MATH" --split "test" --total_num_samples 8

# accelerate launch --num_processes=4 generateAnswers.py --model_name "model/Qwen2.5-Math-7B-Instruct" --dataset_name "aime24" --split "test" --total_num_samples 8
# accelerate launch --num_processes=4 generateAnswers.py --model_name "model/Qwen2.5-Math-7B-Instruct" --dataset_name "amc23" --split "test" --total_num_samples 8
# accelerate launch --num_processes=4 generateAnswers.py --model_name "model/Qwen2.5-Math-7B-Instruct" --dataset_name "olympiad" --split "train" --total_num_samples 8


# accelerate launch --num_processes=4 generateAnswers.py --model_name "model/Qwen2.5-Math-7B-Instruct" --dataset_name "arc" --split "test" --total_num_samples 20