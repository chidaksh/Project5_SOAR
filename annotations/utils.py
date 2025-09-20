from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm

MODEL_ID = "microsoft/Phi-3-mini-128k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONTEXT_LENGTH = 10000

def setup_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def calculate_token_length(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def apply_mid_truncation(context: str, tokenizer, max_length: int = None) -> str:
    if max_length is None:
        max_length = MAX_CONTEXT_LENGTH
    ids = tokenizer.encode(context, add_special_tokens=False)
    if len(ids) <= max_length:
        return context
    keep_head = max_length // 2
    keep_tail = max_length - keep_head
    truncated = ids[:keep_head] + ids[-keep_tail:]
    return tokenizer.decode(truncated, skip_special_tokens=True)

def calculate_log_odds(prob: float, eps: float = 1e-8) -> float:
    prob = max(eps, min(1 - eps, prob))
    return np.log(prob / (1 - prob))

def load_msmarco_dataset(
    query_types=None, 
    total_examples=1000, 
    seed=42,
    only_selected_passages=False
):
    ds = load_dataset("ms_marco", "v2.1", split="train")

    if query_types:
        if isinstance(query_types, str):
            query_types = [query_types]
        qset = set(query_types)
        ds = ds.filter(lambda x: x.get("query_type", "") in qset)

    if len(ds) == 0:
        raise ValueError(
            f"No examples found for query_types={query_types}. "
            "Try different types or pass None to disable filtering."
        )

    if only_selected_passages:
        print("Processing dataset to retain all selected passages and their URLs...")
        processed_rows = []
        for row in tqdm(ds, desc="Retaining All Selected Passages"):
            passages_data = row.get('passages', {})
            passage_texts = passages_data.get('passage_text', [])
            passage_urls = passages_data.get('url', [])
            is_selected_flags = passages_data.get('is_selected', [])
            
            selected_indices = [i for i, flag in enumerate(is_selected_flags) if flag == 1]
            
            if selected_indices:
                selected_passages = [passage_texts[i] for i in selected_indices if i < len(passage_texts)]
                selected_urls = [passage_urls[i] for i in selected_indices if i < len(passage_urls)]

                if selected_passages and len(selected_passages) == len(selected_urls):
                    new_row = row.copy()
                    new_row['passages'] = {
                        'passage_text': selected_passages,
                        'url': selected_urls,
                        'is_selected': [1] * len(selected_passages)
                    }
                    processed_rows.append(new_row)

        if not processed_rows:
            raise ValueError("Processing removed all examples. Check dataset integrity.")
            
        ds = Dataset.from_list(processed_rows)
        print(f"Processing complete. New dataset size: {len(ds)} examples.")
    else:
        print("Proceeding with original, unfiltered passages for each example.")

    k = min(total_examples, len(ds))
    ds = ds.shuffle(seed=seed).select(range(k))
    print(f"Returning a final dataset of {len(ds)} examples.")
    return ds

def format_msmarco_prompt(row: Dict, tokenizer) -> str:
    """
    [CORRECTED VERSION]
    Format MS MARCO data into a prompt for the model.
    This version correctly handles the ms_marco v2.1 data structure.
    """
    query = row.get('query', '')
    
    passages_text = ""
    if 'passages' in row and isinstance(row['passages'], dict) and row['passages'].get('passage_text'):
        # Correctly join all passage texts found in the list.
        passage_list = row['passages']['passage_text']
        passages_text = "\n\n".join(passage_list)
    
    if tokenizer and passages_text and calculate_token_length(passages_text, tokenizer) > MAX_CONTEXT_LENGTH:
        passages_text = apply_mid_truncation(passages_text, tokenizer)
        
    msgs = [
        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the given context."},
        {"role": "user", "content": f"Context:\n{passages_text}\n\nQuestion: {query}\n\nAnswer:"}
    ]
    
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

