#!/usr/bin/env python3
# Copyright (C) 2024 Charles O. Goddard
"""
Preprocess the RedPajama sample dataset for training.

Tokenize the text, filter out examples that compress too poorly or are mostly
numbers, and chunk into equal-sized pieces.

Sorts the chunks by the average token probability in the style of "CRAMMING: TRAINING
A LANGUAGE MODEL ON A SINGLE GPU IN ONE DAY" (https://arxiv.org/abs/2212.14034). Also
filters out chunks with fewer than 10 unique tokens.
"""
from collections import Counter
import datasets
import torch
import tqdm
import transformers

# Constants for configuration
NUM_PROC = 64
MAX_COMPRESSION_RATIO = 0.3
MAX_NUMBER_FRAC = 0.1
CHUNK_SIZE = 4096

# Initialize the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def do_tokenize(examples):
    """
    Tokenize text and compute the compression ratio.
    
    Args:
        examples (dict): A dictionary containing the text to be tokenized.
        
    Returns:
        dict: Tokenized input with added compression ratio.
    """
    tokenized = tokenizer(examples["text"], return_tensors="pt")
    return {
        **tokenized,
        "compression_ratio": tokenized.input_ids.numel() / len(examples["text"]),
    }

def number_frac(examples):
    """
    Estimate the fraction of words that are numbers in the given examples.
    
    Args:
        examples (dict): A dictionary containing the text to be analyzed.
        
    Returns:
        float: Fraction of words that are numbers.
    """
    count = sum(chunk.replace(",", "").replace(".", "").isdigit() for chunk in examples["text"].split())
    total_words = len(examples["text"].split())
    return count / max(1, total_words)

def chunkinate(ds, chunk_size):
    """
    Chop up examples into equal-sized chunks.
    
    Args:
        ds (Dataset): The dataset to be chunked.
        chunk_size (int): Size of each chunk.
        
    Yields:
        dict: A dictionary of chunks with 'input_ids' and 'attention_mask'.
    """
    current_chunk = {"input_ids": [], "attention_mask": []}
    for example in tqdm.tqdm(ds, total=len(ds)):
        ids_in, mask_in = example["input_ids"][0], example["attention_mask"][0]
        if ids_in[-1] != tokenizer.eos_token_id:
            ids_in.append(tokenizer.eos_token_id)
            mask_in.append(1)
        while ids_in:
            free_space = chunk_size - len(current_chunk["input_ids"])
            chunk_length = min(len(ids_in), free_space)
            if chunk_length == 0:
                yield current_chunk
                current_chunk = {"input_ids": [], "attention_mask": []}
                continue
            current_chunk["input_ids"].extend(ids_in[:chunk_length])
            current_chunk["attention_mask"].extend(mask_in[:chunk_length])
            ids_in = [tokenizer.bos_token_id] + ids_in[chunk_length:]
            mask_in = [1] + mask_in[chunk_length:]
    if current_chunk["input_ids"]:
        yield current_chunk

# Load dataset
ds = datasets.load_dataset("togethercomputer/RedPajama-Data-1T-Sample")["train"]
ds_p = ds.map(do_tokenize, num_proc=NUM_PROC).filter(
    lambda ex: ex["compression_ratio"] <= MAX_COMPRESSION_RATIO, num_proc=NUM_PROC
)

# Calculate token probabilities
token_counts = Counter()
for example in tqdm.tqdm(ds_p, total=len(ds_p)):
    token_counts.update(example["input_ids"][0])
tok_probs = torch.tensor(
    [token_counts.get(idx, 0) for idx in range(tokenizer.vocab_size)]
) / sum(token_counts.values())

def example_avg_token_prob(example):
    """
    Calculate the average token probability for an example.
    
    Args:
        example (dict): A dictionary containing 'input_ids'.
        
    Returns:
        torch.Tensor: The average token probability.
    """
    return torch.mean(tok_probs[example["input_ids"][0][:2048]])

# Further preprocessing
ds_pp = ds_p.map(
    lambda ex: {"avg_token_prob": example_avg_token_prob(ex)}, num_proc=NUM_PROC
)
ds_ppp = ds_pp.filter(lambda ex: number_frac(ex) < MAX_NUMBER_FRAC, num_proc=NUM_PROC)
ds_pppp = ds_ppp.sort(["avg_token_prob"], reverse=True)

def unique_tokens(example):
    """
    Count unique tokens in an example.
    
    Args:
        example (dict): A dictionary containing 'input_ids'.
        
    Returns:
        int: Number of unique tokens.
    """
    return len(set(example["input_ids"]))

# Final chunking and filtering
chunked = datasets.Dataset.from_generator(lambda: chunkinate(ds_pppp, CHUNK_SIZE))
chunked = chunked.filter(lambda ex: unique_tokens(ex) > 10, num_proc=NUM_PROC)
chunked.save_to_disk(f"redpajama_1t_{CHUNK_SIZE}")
