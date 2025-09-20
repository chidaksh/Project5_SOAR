from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from datasets import load_dataset
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
        torch_dtype=torch.bfloat16,
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

def build_mcq_user_message(row: Dict, tokenizer=None, context=None) -> str:
    choices = []
    for key in ["choice_A", "choice_B", "choice_C", "choice_D"]:
        if key in row and row[key] is not None and row[key] != "":
            choices.append(key[-1] + ". " + str(row[key]))
    choices_text = "\n".join(choices) if choices else "A.\nB.\nC.\nD."

    if context is None:
        ctx = row.get("context", "")
    else:
        ctx = context
    
    if tokenizer and calculate_token_length(ctx, tokenizer) > MAX_CONTEXT_LENGTH:
        ctx = apply_mid_truncation(ctx, tokenizer)
    
    ques = row.get("question", "")

    user_msg = (
        f"{ctx}\n\n"
        "=== QUESTION ===\n"
        f"{ques}\n\n"
        "=== CHOICES ===\n"
        f"{choices_text}\n\n"
        "=== FORMAT ===\n"
        "Final answer: "
    )
    return user_msg

def build_chat_prompt(row: Dict, tokenizer) -> str:
    system_message = {"role": "system", "content": "You are a careful, concise MCQ solver. Respond with A, B, C, or D only, no addtional text or explanation."}
    user_message_scaffold = build_mcq_user_message(row, tokenizer, context="")
    messages_scaffold = [system_message, {"role": "user", "content": user_message_scaffold}]
    prompt_scaffold_string = tokenizer.apply_chat_template(messages_scaffold, tokenize=False, add_generation_prompt=True)
    overhead_tokens = calculate_token_length(prompt_scaffold_string, tokenizer)
    
    safety_margin = 50 
    context_token_budget = MAX_CONTEXT_LENGTH - overhead_tokens - safety_margin
    
    if context_token_budget < 0:
        print("WARNING: Question and choices alone exceed MAX_CONTEXT_LENGTH. Context will be empty.")
        context_token_budget = 0
    original_context = row.get("context", "")
    truncated_context = apply_mid_truncation(original_context, tokenizer, context_token_budget)
    final_user_message = build_mcq_user_message(row, tokenizer, context=truncated_context)
    final_messages = [system_message, {"role": "user", "content": final_user_message}]
    final_prompt = tokenizer.apply_chat_template(final_messages, tokenize=False, add_generation_prompt=True)

    final_length = calculate_token_length(final_prompt, tokenizer)
    if final_length > MAX_CONTEXT_LENGTH:
        print(f"WARNING: Final prompt length ({final_length}) still exceeds max length after truncation.")

    return final_prompt

def build_prompt_with_context(row: Dict, new_context: str, tokenizer) -> str:
    r = dict(row)
    r["context"] = new_context
    return build_chat_prompt(r, tokenizer)

def get_choice_variants(choice: str) -> List[str]:
    return [choice, f" {choice}", f"\n{choice}"]

def first_step_choice_probs(
    model, tokenizer, prompt: str, choices: Tuple[str, ...] = ("A", "B", "C", "D")
) -> Tuple[Dict[str, float], torch.Tensor, List[int]]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
    
    probs_dict = {}
    choice_ids = []
    
    for choice in choices:
        variants = get_choice_variants(choice)
        max_prob = 0.0
        best_token_id = None
        
        for variant in variants:
            tokens = tokenizer.encode(variant, add_special_tokens=False)
            if tokens:
                token_id = tokens[0]
                token_prob = probs[0, token_id].item()
                if token_prob > max_prob:
                    max_prob = token_prob
                    best_token_id = token_id
        
        if best_token_id is not None:
            probs_dict[choice] = max_prob
            choice_ids.append(best_token_id)
        else:
            probs_dict[choice] = 0.0
            choice_ids.append(tokenizer.unk_token_id)
    
    del inputs, outputs, logits
    torch.cuda.empty_cache()
    
    return probs_dict, probs[0].detach().cpu(), choice_ids

def greedy_inference(model, tokenizer, prompt: str, max_new_tokens=3) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    
    new_tokens = generated_ids[0][inputs['input_ids'].shape[1]:]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    del inputs, generated_ids, new_tokens
    torch.cuda.empty_cache()
    
    return output

def get_token_probability_from_input_ids(model, tokenizer, input_ids: torch.Tensor, target_choice: str) -> float:
    """Get token probability directly from input_ids tensor with memory optimization."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    # Ensure input_ids is on the correct device
    if input_ids.device != model.device:
        input_ids = input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
    
    choice_variants = get_choice_variants(target_choice)
    choice_probs = []
    
    for variant in choice_variants:
        choice_tokens = tokenizer.encode(variant, add_special_tokens=False)
        if choice_tokens:
            choice_id = choice_tokens[0]
            choice_probs.append(probs[0, choice_id].item())
        else:
            choice_probs.append(0.0)
    
    max_prob = max(choice_probs) if choice_probs else 0.0
    
    # Clean up tensors
    del outputs, logits, probs
    torch.cuda.empty_cache()
    
    return max_prob

def get_token_probability(model, tokenizer, prompt: str, target_choice: str) -> float:
    probs_dict, full_probs, _ = first_step_choice_probs(model, tokenizer, prompt, choices=(target_choice,))
        
    del full_probs
    torch.cuda.empty_cache()
    
    return probs_dict.get(target_choice, 0.0)


def calculate_log_odds(prob: float, eps: float = 1e-8) -> float:
    prob = max(eps, min(1 - eps, prob))
    return np.log(prob / (1 - prob))

def load_longbench_dataset():
    return load_dataset('THUDM/LongBench-v2', split='train')

# def load_msmarco_dataset(query_types=None, total_examples=1000, seed=42):
#     """
#     Load MS MARCO dataset training split with filtering by query_type.
#     """
#     import random
#     from datasets import Dataset
    
#     random.seed(seed)
#     full_dataset = load_dataset('ms_marco', 'v2.1', split='train')
    
#     if query_types is None:
#         query_types = ['numeric']
    
#     print(f"Filtering dataset for query types: {query_types}")
#     print(f"Dataset size: {len(full_dataset)}")
    
#     # First, examine actual query_type values in the dataset
#     query_type_counts = {}
#     sample_size = len(full_dataset)
    
#     for i in range(sample_size):
#         example = full_dataset[i]
#         qt = example.get('query_type', 'unknown')
#         query_type_counts[qt] = query_type_counts.get(qt, 0) + 1
    
#     print("Available query_type values in dataset:")
#     for qt, count in sorted(query_type_counts.items()):
#         print(f"  {qt}: {count}")
    
#     # Optimized: Filter dataset by query types first, then sample
#     examples_per_type = total_examples // len(query_types)
#     print(f"Target: {examples_per_type} examples per type")
    
#     all_examples = []
#     for qt in query_types:
#         # Filter dataset for current query type
#         type_filtered = full_dataset.filter(lambda x: x.get('query_type', '') == qt)
        
#         if len(type_filtered) == 0:
#             print(f"Warning: No examples found for query_type '{qt}'")
#             continue
            
#         # Sample required number from filtered subset
#         sample_size = min(examples_per_type, len(type_filtered))
#         if sample_size < len(type_filtered):
#             sampled_indices = random.sample(range(len(type_filtered)), sample_size)
#             sampled_examples = type_filtered.select(sampled_indices)
#         else:
#             sampled_examples = type_filtered
            
#         print(f"Found {len(type_filtered)} total, sampled {len(sampled_examples)} examples for query_type '{qt}'")
#         all_examples.extend(sampled_examples)
    
#     if all_examples:
#         random.shuffle(all_examples)
#         return Dataset.from_list(all_examples)
#     else:
#         print("Warning: No examples found. Using random subset.")
#         indices = random.sample(range(len(full_dataset)), min(total_examples, len(full_dataset)))
#         return full_dataset.select(indices)
def load_msmarco_dataset(query_types=None, total_examples=1000, seed=42,
                         only_selected_passages=False):
    """
    Load MS MARCO v2.1 train split, optionally filter by query_type(s),
    and return up to `total_examples` randomly sampled rows.
    """
    from datasets import load_dataset, Dataset

    # Load
    ds = load_dataset("ms_marco", "v2.1", split="train")

    # Optional filter by query_type
    if query_types:
        if isinstance(query_types, str):
            query_types = [query_types]
        qset = set(query_types)
        ds = ds.filter(lambda x: x.get("query_type", "") in qset)

    # Guard: nothing matched
    if len(ds) == 0:
        raise ValueError(
            f"No examples found for query_types={query_types}. "
            "Try different types or pass None to disable filtering."
        )

    # --- MAIN LOGIC BLOCK ---
    if only_selected_passages:
        print("Processing dataset to retain all selected passages and their URLs...")
        processed_rows = []
        
        for row in tqdm(ds, desc="Retaining All Selected Passages"):
            passages_data = row.get('passages', {})
            passage_texts = passages_data.get('passage_text', [])
            passage_urls = passages_data.get('url', []) # Get the list of URLs
            is_selected_flags = passages_data.get('is_selected', [])
            
            # Find ALL indices where the flag is 1
            selected_indices = [i for i, flag in enumerate(is_selected_flags) if flag == 1]
            
            if selected_indices:
                # Collect all passages and URLs corresponding to the selected indices
                # Includes safety checks for misaligned data
                selected_passages = [
                    passage_texts[i] 
                    for i in selected_indices 
                    if i < len(passage_texts)
                ]
                selected_urls = [
                    passage_urls[i] 
                    for i in selected_indices 
                    if i < len(passage_urls)
                ]

                # Ensure we have a matching number of passages and URLs
                if selected_passages and len(selected_passages) == len(selected_urls):
                    new_row = row.copy()
                    new_row['passages'] = {
                        'passage_text': selected_passages,
                        'url': selected_urls, # Add the list of selected URLs
                        'is_selected': [1] * len(selected_passages)
                    }
                    processed_rows.append(new_row)

        if not processed_rows:
            raise ValueError("Processing removed all examples. Check dataset integrity.")
            
        # Overwrite 'ds' with the new, processed dataset
        ds = Dataset.from_list(processed_rows)
        print(f"Processing complete. New dataset size: {len(ds)} examples.")
    
    # --- The 'else' case is simply doing nothing to the passages ---
    else:
        print("Proceeding with original, unfiltered passages for each example.")
        pass # No processing needed

    # Sample deterministically
    k = min(total_examples, len(ds))
    ds = ds.shuffle(seed=seed).select(range(k))
    return ds

def format_msmarco_prompt(row: Dict, tokenizer) -> str:
    """Format MS MARCO data into a prompt for the model."""
    # MS MARCO dataset structure varies by version, let's handle multiple cases
    
    # Try different possible field names
    query = row.get('query', row.get('question', ''))
    
    # Handle different passage formats
    passages_text = ""
    if 'passages' in row and row['passages']:
        passages = row['passages']
        if isinstance(passages, list):
            # Format: list of passage objects
            passages_text = "\n\n".join([
                f"Passage {i+1}: {p.get('passage_text', p.get('text', str(p)))}" 
                for i, p in enumerate(passages[:5])  # Limit to first 5 passages
            ])
        elif isinstance(passages, dict):
            # Format: dict with passage info
            for key in ['passage_text', 'text', 'content']:
                if key in passages:
                    passages_text = f"Passage: {passages[key]}"
                    break
    elif 'context' in row:
        # Alternative format with context field
        passages_text = f"Context: {row['context']}"
    elif 'wellFormedAnswers' in row and row['wellFormedAnswers']:
        # Some MS MARCO versions have answers, use those as context
        passages_text = f"Context: {' '.join(row['wellFormedAnswers'][:3])}"
    
    # Fallback: create a simple prompt without context
    if not passages_text and not query:
        return "Please provide a helpful response."
    
    # Apply truncation if needed
    if tokenizer and passages_text and calculate_token_length(passages_text, tokenizer) > MAX_CONTEXT_LENGTH:
        passages_text = apply_mid_truncation(passages_text, tokenizer)
    
    # Create a QA prompt
    if passages_text:
        msgs = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question based on the given passages."},
            {"role": "user", "content": f"{passages_text}\n\nQuestion: {query}\n\nAnswer:"}
        ]
    else:
        # Fallback without passages
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Question: {query}\n\nAnswer:"}
        ]
    
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def pick_reasonable_row(ds, index) -> Dict:
    return ds[index]
