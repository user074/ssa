# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from xml.etree import ElementTree as ET

import torch

from torchtune.modules.transforms.tokenizers import ModelTokenizer
from torchtune.models.llama3 import Llama3Tokenizer


def extract_tags_eval(text: str) -> dict[str, list[str]]:
    """
    Parse XML-like tags from text. Returns a dictionary with keys 'think' and 'answer'.
    The values are lists of strings, with each string being the content of a tag.
    """
    xml_string = f"<root>{text}</root>"
    root = ET.fromstring(xml_string)

    return {
        "think": [
            elem.text if elem.text is not None else "" for elem in root.findall("think")
        ],
        "answer": [
            elem.text if elem.text is not None else ""
            for elem in root.findall("answer")
        ],
        "evaluate": [
            elem.text if elem.text is not None else ""
            for elem in root.findall("evaluate")
        ],
        "verify": [
            elem.text if elem.text is not None else ""
            for elem in root.findall("verify")
        ],
    }


def shaped_correctness_reward_eval(answer: str, completion: str) -> tuple[float, float]:
    """
    Reward function for verifiable rewards with some mild shaping.
    In evaluation for selfplay, we want to reward the model for the following:
    - Correct answer and wrong verification
    - Incorrect answer and correct verification
    

    Args:
        answer (str): ground-truth answer to the current problem
        completion (str): model's completion, starting immediately after "Assistant: <think>"
    Returns:
        reward: (float) a shaped reward indicating the correct answer and the correct format
        success: (float) a binary measure of success (1 if the answer is correct and correctly formatted, 0 otherwise)
    """
    eval_reward = 0.0
    gen_reward = 0.0
    success = 0.0
    gen_success = 0.0
    eval_success = 0.0
    both_success = 0.0
    # print("completion", completion)
    # print("answer", answer)
    try:
        tags = extract_tags_eval("<think>" + completion.replace("<<", "").replace(">>", ""))
    except ET.ParseError:
        tags = {"think": [], "answer": [], "evaluate": [], "verify": []}

    # FORMAT REWARD: encourage proper formatting explicitly
    # format_reward = 0.0
    gen_reward += 5.0 if len(tags["answer"]) == 1 else 0.0
    gen_reward += 5.0 if len(tags["think"]) == 1 else 0.0
    eval_reward += 5.0 if len(tags["evaluate"]) == 3 else 0.0
    eval_reward += 5.0 if len(tags["verify"]) == 3 else 0.0
    # reward += format_reward
    # evaluate_ans = ""
    # verify_ans = ""
    # if len(tags["evaluate"]) > 2:
    #     evaluate_ans = tags["evaluate"][-1].strip()
        
    # if len(tags["evaluate"]) == 2 or evaluate_ans == "":
    #     format_reward -= 20.0
    # elif len(tags["evaluate"]) == 3:
    #     format_reward += 5.0
    
    # if len(tags["verify"]) > 2:
    #     verify_ans = tags["verify"][-1].strip()
    
    # if len(tags["verify"]) == 2 or verify_ans == "":
    #     format_reward -= 20.0
    # elif len(tags["verify"]) == 3:
    #     format_reward += 5.0
    
    # PENALTY FOR EMPTY ANSWER AND THINK
    # if len(tags["answer"]) == 0 or tags["answer"][0].strip() == "":
    #     format_reward -= 20.0  # explicit penalty for empty or missing answe
    # else:
    #     format_reward += 5.0

    # if len(tags["think"]) == 0 or tags["think"][0].strip() == "":
    #     format_reward -= 20.0  # explicit penalty for empty reasoning
    # else:
    #     format_reward += min(5.0, len(tags["think"][0]) * 0.01) if len(tags["think"]) > 0 else 0.0
    # format_reward += min(5.0, len(tags["answer"][0]) * 0.01) if len(tags["answer"]) > 0 else 0.0
    
    # PENALTY FOR Repeated Tokens in think
    # tokens = tags["think"][0].split() if len(tags["think"]) > 0 else []
    # if len(tokens) > 0:
    #     unique_tokens = set(tokens)
    #     diversity = len(unique_tokens) / (len(tokens) + 1e-5)
    #     if diversity < 0.2:
    #         format_reward -= 20.0 * (0.2 - diversity)
    

   # ADVERSARIAL REWARD (Stage 1 logic explicitly correct):
    # Evaluator: +reward if catches incorrect answer
    # Generator: -reward if successfully fools evaluator
    
    # answer_correct = len(tags["answer"]) > 0 and tags["answer"][-1].strip() == answer.strip()
    # verification = tags["verify"][-1].lower() if len(tags["verify"]) > 2 else ""

    # if answer_correct and "wrong" == verification:
    #     # Evaluator wrongly rejected correct answer (Evaluator fooled by Generator)
    #     reward = 0.0
    #     reward += format_reward
    # elif not answer_correct and "correct" == verification:
    #     # Evaluator wrongly accepted incorrect answer (Evaluator fooled by Generator)
    #     # reward -= 100.0
    #     reward -= 100.0 # Set to 0 for 1st stage
    #     reward += format_reward
    # elif not answer_correct and "wrong" == verification:
    #     # Evaluator correctly rejected incorrect answer (Evaluator succeeded)
    #     reward += 100.0
    #     reward += format_reward
    # elif answer_correct and "correct" == verification:
    #     # Evaluator correctly accepted correct answer (Evaluator succeeded)
    #     reward += 50.0 # Set to 0 for 1st stage
    #     reward += format_reward
    # if answer_correct:
    #     success = 1.0  # explicitly successful scenario

    #Separate rewards for generator and evaluator
    verification = tags["verify"][-1].lower() if len(tags["verify"]) > 2 else ""
    #ensure the verification is in the right format
    if verification == "correct" or verification == "wrong":
        eval_reward += 10.0
        
    # if any(attempt == answer for attempt in tags["answer"]):
    #     # One of the answer tags has the right answer
    #     gen_reward += 20.0
    #     if verification == "correct":
    #         eval_reward += 20.0

    # if any((answer in attempt) for attempt in tags["answer"]):
    #     # One of the answer tags contains the right answer (might be e.g. $20 instead of 20)
    #     gen_reward += 20.0
    #     if verification == "correct":
    #         eval_reward += 20.0
    

    if len(tags["answer"]) > 0 and tags["answer"][0] == answer:
        #Generator Correct Answer
        if verification == "correct" or ('correct' in verification and 'wrong' not in verification):
            #Evaluator Correct Answer
            gen_reward += 100.0
            eval_reward += 100.0
            both_success = 1.0
        elif verification == "wrong" or ('wrong' in verification and 'correct' not in verification):
            #Generator Incorrect Answer
            gen_reward += 50.0
            eval_reward = 0.0
            gen_success = 1.0
        else:
            #Evaluator Incorrect Format
            gen_reward += 50.0
            eval_reward = 0.0
            gen_success = 1.0
        success = 1
    elif len(tags["answer"]) > 0 and answer in tags["answer"][0]:
        #Generator Correct Answer
        #Generator Correct Answer
        if verification == "correct" or ('correct' in verification and 'wrong' not in verification and 'incorrect' not in verification):
            #Evaluator Correct Answer
            gen_reward += 50.0
            eval_reward += 100.0
            both_success = 1.0
        elif verification == "wrong" or ('wrong' in verification and 'correct' not in verification):
            #Generator Incorrect Answer
            gen_reward += 20.0
            eval_reward = 0.0
            gen_success = 1.0
        else:
            #Evaluator Incorrect Format
            gen_reward += 20.0
            eval_reward = 0.0
            gen_success = 1.0
        success = 1
    
    elif len(tags["answer"]) > 0 and tags["answer"][0] != answer:
        #Generator Incorrect Answer
        #Might need to add if correct is in the verify or not
        if verification == "correct" or ('correct' in verification and 'wrong' not in verification and 'incorrect' not in verification):
            #Evaluator Correct Answer 
            gen_reward += 0.0
            eval_reward -= 100.0
        elif verification == "wrong" or ('wrong' in verification and 'correct' not in verification):
            # gen_reward = 0.0
            gen_reward += 0.0
            eval_reward +=  100.0
            eval_success = 1.0
        else:
            #Evaluator Incorrect Format
            eval_reward -= 100.0
            
    #Evaluator just adversarial attack and create </answer> tag...
    if len(tags["answer"]) == 0 or len(tags["think"]) == 0 or len(tags["evaluate"]) == 0 or len(tags["verify"]) == 0:
        eval_reward = -100.0
        gen_reward = -100.0
    
    return gen_reward, eval_reward, success, gen_success, eval_success, both_success


def batch_shaped_evaluation_correctness_reward(
    tokenizer: ModelTokenizer, completions: torch.Tensor, answers: list[str]
) -> [torch.Tensor, torch.Tensor]:
    """Utility function to apply the shaped reward function to a GRPO-style batch of completions."""

    batch_size, grpo_size, *_ = completions.shape
    gen_rewards = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
    eval_rewards = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
    successes = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
    gen_successes = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
    eval_successes = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
    both_successes = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
    # completions :: [B, G, L]
    for b in range(batch_size):
        for g in range(grpo_size):
            # Replace invalid tokens with pad_id. Monkey patch for now.
            valid_tokens = completions[b, g].clone()
            # valid_tokens[valid_tokens >= tokenizer.vocab_size] = tokenizer.pad_id
            #Check if it is llama3b tokenizer
            if isinstance(tokenizer, Llama3Tokenizer):
                valid_tokens[valid_tokens == 128011] = tokenizer.pad_id
                text_completion = tokenizer.decode(valid_tokens.tolist())
            else:
                text_completion = tokenizer.decode(valid_tokens.tolist(), skip_special_tokens = True)
            # print("text_completion", text_completion)
            # print("answers", answers[b])
            gen_reward, eval_reward, success, gen_success, eval_success, both_success = shaped_correctness_reward_eval(
                answer=answers[b], completion=text_completion
            )
            gen_rewards[b, g] = gen_reward
            eval_rewards[b, g] = eval_reward
            successes[b, g] = success
            gen_successes[b, g] = gen_success
            eval_successes[b, g] = eval_success
            both_successes[b, g] = both_success
    return gen_rewards, eval_rewards, successes, gen_successes, eval_successes, both_successes
