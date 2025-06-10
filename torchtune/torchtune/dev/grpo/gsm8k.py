# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import re
from typing import Any, Callable, Dict, Optional

from torchtune.datasets import SFTDataset
from torchtune.modules.tokenizers import ModelTokenizer

from .data import ReasoningProblem, RLDataset

# TODO: dedup this between here and _rl
PREAMBLE_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, "
    "i.e., <think>reasoning process here</think> <answer>answer here</answer>. User: {question} Assistant: "
)

TRAINABLE_PROMPT = "<think>{cot}</think> <answer>{answer}</answer>"

# Prompts for concatenated data transformation
CONCAT_BASE_PROMPT = (
    "A conversation between User and Assistant. The user provide a question and some proposed answers. The Assistant first evaluate each answers individually,check whether each answer directly addresses the original question, assess the correctness of each answer based on logical reasoning, calculations, and accuracy relative to the question. After thorough evaluation, identify one correct answer. If the correct answer is not in the provided proposed answers, the Assistant will combine the correct answer with the proposed answers and provide the correct answer."
    "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, "
    "i.e., <think>reasoning process here</think> <answer>answer here</answer>. User: {}."
)

CONCAT_TRAINABLE_PROMPT = "Assistant: {}"

def normalize_gsm(problem: dict[str, str]) -> ReasoningProblem:
    """
    Parses an item from the GSM8K dataset into a ReasoningProblem by splitting it up into the question, cot, and answer.
    """
    question = problem["question"]
    solution = problem["answer"]

    cot, answer = solution.split("#### ")

    return {"question": question, "cot": cot, "answer": answer}


def sft_gsm_transform(problem: dict[str, str]) -> dict[str, str]:
    """
    Prepares an item from the GSM8k into a format that can be used for SFT.
    """
    question = problem["question"]
    solution = problem["answer"]

    cot, answer = solution.split("#### ")

    preamble = PREAMBLE_PROMPT.format(question=question)
    trainable = TRAINABLE_PROMPT.format(cot=cot, answer=answer)

    return {"preamble": preamble, "trainable": trainable}


def gsm8k_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "openai/gsm8k",
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    name: str = "main",
    partition: Optional[str] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> RLDataset:
    """
    GSM8k dataset from OpenAI, prepared for RL-based training with verifiable rewards.
    """

    def default_filter_fn(example: dict, idx: int):
        if partition is None:
            return True

        match = re.match(r"^(\d+)-(\d+)/(\d+)$", partition)
        if not match:
            raise ValueError(
                f"Invalid partition format: {partition}. Expected format: start-end/total"
            )

        start, end, total = map(int, match.groups())

        current = idx % total
        return start <= current <= end

    filter_fn = filter_fn if filter_fn is not None else default_filter_fn

    ds = RLDataset(
        source=source,
        name=name,
        tokenizer=tokenizer,
        problem_transform=normalize_gsm,
        filter_fn=filter_fn,
        filter_kwargs=dict(with_indices=True),
        split=split,
        **load_dataset_kwargs,
    )

    return ds


def gsm8k_sft(
    tokenizer: ModelTokenizer,
    *,
    source: str = "openai/gsm8k",
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    name: str = "main",
    partition: Optional[str] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> SFTDataset:
    """
    GSM8k dataset from OpenAI, prepared for SFT-based training with CoT.
    """

    def model_transform(problem: dict[str, str]) -> dict[str, list[int]]:
        pre_tokens = tokenizer.encode(problem["preamble"], add_eos=False)
        trainable_tokens = tokenizer.encode(problem["trainable"], add_bos=False)

        # 1 == discard the token, 0 == include the token in training
        mask = [1 for t in pre_tokens] + [0 for t in trainable_tokens]

        return {"tokens": pre_tokens + trainable_tokens, "mask": mask}

    def default_filter_fn(example: dict, idx: int):
        if partition is None:
            return True

        match = re.match(r"^(\d+)-(\d+)/(\d+)$", partition)
        if not match:
            raise ValueError(
                f"Invalid partition format: {partition}. Expected format: start-end/total"
            )

        start, end, total = map(int, match.groups())

        current = idx % total
        return start <= current <= end

    filter_fn = filter_fn if filter_fn is not None else default_filter_fn

    ds = SFTDataset(
        source=source,
        message_transform=sft_gsm_transform,
        model_transform=model_transform,
        filter_fn=filter_fn,
        filter_kwargs=dict(with_indices=True),
        split=split,
        name=name,
        **load_dataset_kwargs,
    )

    return ds

def get_ground_truth_gsm8k_answer(answer: str) -> str:
    """
    Extracts and formats the ground truth answer from GSM8K format.
    Args:
        answer (str): The answer string in GSM8K format (with #### separator)
    Returns:
        str: Formatted answer with reasoning and final answer
    """

    return CONCAT_TRAINABLE_PROMPT.format(answer)

def transform_concat_gsm(question: str) -> str:
    """
    Concatenates the question and answer with the base prompt format.
    Args:
        question (str): The question text
        answer (str): The answer text
    Returns:
        str: Formatted prompt with question and answer
    """
    return CONCAT_BASE_PROMPT.format(question)

def concat_gsm_transform(problem: dict[str, str]) -> dict[str, str]:
    """
    Transforms a GSM8K problem into a format suitable for concatenated SFT training.
    Args:
        problem (dict): Dictionary containing 'question' and 'answer' keys
    Returns:
        dict: Dictionary with formatted prompt and answer
    """
    question = problem["question"]
    answer = problem["model_response"]
    
    # Create the final prompt
    preamble = transform_concat_gsm(question)
    # Get formatted ground truth answer
    trainable = get_ground_truth_gsm8k_answer(answer)
    
    return {"preamble": preamble, "trainable": trainable}

def gsm8k_concat_sft(
    tokenizer: ModelTokenizer,
    *,
    source: str = "",
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    name: str = None,
    partition: Optional[str] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> SFTDataset:
    """
    GSM8k dataset from OpenAI, prepared for SFT-based training with concatenated answers.
    """
    def model_transform(problem: dict[str, str]) -> dict[str, list[int]]:
        pre_tokens = tokenizer.encode(problem["preamble"], add_eos=False)
        trainable_tokens = tokenizer.encode(problem["trainable"], add_bos=False, add_eos=False) #manually add eos token since default is im_end which is not the actual eos token
        #actual eos token is pad_id. Monkey Patch for QWEN. This needs to be fixed for other models
        #Issue: https://github.com/pytorch/torchtune/issues/2587
        trainable_tokens.append(tokenizer.pad_id)
        
        # 1 == discard the token, 0 == include the token in training
        mask = [1 for t in pre_tokens] + [0 for t in trainable_tokens]

        return {"tokens": pre_tokens + trainable_tokens, "mask": mask}

    def default_filter_fn(example: dict, idx: int):
        if partition is None:
            return True

        match = re.match(r"^(\d+)-(\d+)/(\d+)$", partition)
        if not match:
            raise ValueError(
                f"Invalid partition format: {partition}. Expected format: start-end/total"
            )

        start, end, total = map(int, match.groups())

        current = idx % total
        return start <= current <= end

    filter_fn = filter_fn if filter_fn is not None else default_filter_fn

    ds = SFTDataset(
        source=source,
        message_transform=concat_gsm_transform,
        model_transform=model_transform,
        filter_fn=filter_fn,
        filter_kwargs=dict(with_indices=True),
        split=split,
        name=name,
        **load_dataset_kwargs,
    )

    return ds
