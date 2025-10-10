import json
import os
import torch
import numpy as np
import gc
from tqdm import tqdm
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

RANDOM_SEED = 42
MODEL_ID = "ministral/Ministral-3b-instruct"
model_key = 'mistral3'
MAX_CONTEXT_LENGTH = 4000
DATASET_NAME = "ms_marco"
DATASET_CONFIG = "v1.1"
QUERY_TYPES = ["NUMERIC"]
NUM_SAMPLES = 500
BATCH_SIZE = 32
OUTPUT_FILE = f"passage_attributions_{model_key}.json"
CHECKPOINT_FILE = f"passage_attribution_checkpoint_{model_key}.json"
CHECKPOINT_FREQ = 100

def set_seed(seed=RANDOM_SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.float16, attn_implementation="flash_attention_2", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

def load_dataset_filtered():
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split='test')
    if QUERY_TYPES:
        query_set = set(qt.lower() for qt in QUERY_TYPES)
        dataset = dataset.filter(lambda x: x.get('query_type', '').lower() in query_set)
    k = min(NUM_SAMPLES, len(dataset))
    dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(k))
    return dataset

def create_prompt(context: str, question: str) -> str:
    return f"Using only the information from the context provided, answer the following query.\n\nContext: {context}\n\nQuery: {question}\n\nAnswer:"

def get_log_prob_of_answer(prompts: List[str], answers: List[str], model, tokenizer) -> np.ndarray:
    if not prompts or not answers:
        return np.array([])
    full_texts = [p + " " + a for p, a in zip(prompts, answers)]
    
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONTEXT_LENGTH).to(model.device)
    prompt_inputs = tokenizer(prompts, padding=False, truncation=True, max_length=MAX_CONTEXT_LENGTH)
    prompt_lengths = [len(ids) for ids in prompt_inputs.input_ids]
    
    labels = inputs.input_ids.clone()
    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = -100
    labels[labels == tokenizer.pad_token_id] = -100
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(shift_labels.size())
        per_sample_nll = per_token_loss.sum(dim=1)
        log_probs = -per_sample_nll
    del inputs, labels, outputs
    torch.cuda.empty_cache()
    return log_probs.cpu().numpy()


def extract_msmarco_data(sample: Dict) -> Tuple[str, List[str], str, List[int]]:
    question = sample.get('query', '')
    passages = []
    is_selected = []
    if 'passages' in sample and sample['passages']:
        passage_data = sample['passages']
        if isinstance(passage_data, dict) and 'passage_text' in passage_data:
            passages = [p.strip() for p in passage_data['passage_text'] if p and p.strip()]
        if isinstance(passage_data, dict) and 'is_selected' in passage_data:
            is_selected = passage_data['is_selected']
    answer = ""
    if 'answers' in sample and sample['answers']:
        answer = sample['answers'][0] if isinstance(sample['answers'], list) else sample['answers']
    elif 'wellFormedAnswers' in sample and sample['wellFormedAnswers']:
        answer = sample['wellFormedAnswers'][0]
    return question, passages, answer, is_selected

def analyze_sample(sample: Dict, model, tokenizer) -> Dict:
    question, passages, answer, is_selected = extract_msmarco_data(sample)
    if not passages or not question or not answer:
        return None
    
    if len(passages) <= 1:
        return None
    
    if len(is_selected) != len(passages):
        return None
    
    full_context = "\n\n".join(passages)
    all_prompts = []
    baseline_prompt = create_prompt(full_context, question)
    all_prompts.append(baseline_prompt)
    no_context_prompt = create_prompt("", question)
    all_prompts.append(no_context_prompt)
    for j in range(len(passages)):
        ablated_passages = passages[:j] + passages[j+1:]
        ablated_context = "\n\n".join(ablated_passages)
        all_prompts.append(create_prompt(ablated_context, question))
    all_answers = [answer] * len(all_prompts)
    all_log_probs = []
    for i in range(0, len(all_prompts), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(all_prompts))
        batch_prompts = all_prompts[i:batch_end]
        batch_answers = all_answers[i:batch_end]
        batch_log_probs = get_log_prob_of_answer(batch_prompts, batch_answers, model, tokenizer)
        all_log_probs.extend(batch_log_probs)
    
    baseline_log_prob = float(all_log_probs[0])
    no_context_log_prob = float(all_log_probs[1])
    
    delta_total = baseline_log_prob - no_context_log_prob
    passage_deltas = []
    for j in range(len(passages)):
        ablated_log_prob = float(all_log_probs[2 + j])
        delta = baseline_log_prob - ablated_log_prob
        normalized_delta = delta / delta_total if abs(delta_total) > 1e-9 else 0.0
        passage_deltas.append({
            "passage_idx": j,
            "passage": passages[j],
            "delta_log_prob": delta,
            "normalized_delta": normalized_delta,
            "ablated_log_prob": ablated_log_prob,
            "is_selected": is_selected[j]
        })
    
    del all_log_probs, all_prompts
    torch.cuda.empty_cache()
    
    return {
        "query_id": sample.get('query_id', ''),
        "question": question,
        "full_context": full_context,
        "answer": answer,
        "num_passages": len(passages),
        "baseline_log_prob": baseline_log_prob,
        "no_context_log_prob": no_context_log_prob,
        "delta_total": delta_total,
        "passage_analysis": passage_deltas
    }

def save_checkpoint(results, checkpoint_file):
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)

def load_checkpoint(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        return json.load(f)

def main():
    set_seed()
    model, tokenizer = setup_model_and_tokenizer()
    dataset = load_dataset_filtered()
    
    if os.path.exists(CHECKPOINT_FILE):
        all_results = load_checkpoint(CHECKPOINT_FILE)
    else:
        all_results = []
    processed_ids = {r.get('query_id', '') for r in all_results}
    start_idx = len(all_results)
    
    if start_idx > 0:
        print(f"\nResuming from checkpoint: {start_idx} samples already processed")
    
    all_deltas = []
    max_prompt_len = 0
    max_context_len = 0
    skipped = 0
    errors = 0
    
    for i in tqdm(range(start_idx, len(dataset)), desc="Processing", initial=start_idx, total=len(dataset)):
        sample = dataset[i]
        query_id = sample.get('query_id', '')
        if query_id in processed_ids:
            continue
        
        result = analyze_sample(sample, model, tokenizer)
        if result:
            all_results.append(result)
            processed_ids.add(query_id)
            for p in result['passage_analysis']:
                all_deltas.append(p['delta_log_prob'])
            
            prompt = create_prompt(result['full_context'], result['question'])
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
            context_tokens = len(tokenizer.encode(result['full_context'], add_special_tokens=False))
            max_prompt_len = max(max_prompt_len, prompt_tokens)
            max_context_len = max(max_context_len, context_tokens)
        else:
            skipped += 1
        
        if (i + 1) % CHECKPOINT_FREQ == 0:
            save_checkpoint(all_results, CHECKPOINT_FILE)
            print(f"\nCheckpoint saved at sample {i+1}")
        
        if i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFinal results saved to {OUTPUT_FILE}")
    
    if all_deltas:
        mean_delta = np.mean(all_deltas)
        std_delta = np.std(all_deltas)
        threshold = mean_delta + 2 * std_delta
        print(f"\nProcessed: {len(all_results)} samples")
        print(f"Skipped (<=1 passage): {skipped} samples")
        print(f"Errors: {errors} samples")
        print(f"Total passages analyzed: {len(all_deltas)}")
        print(f"\nContext length analysis:")
        print(f"Max context length: {max_context_len} tokens")
        print(f"Max prompt length: {max_prompt_len} tokens")
        print(f"Truncation threshold: {MAX_CONTEXT_LENGTH} tokens")
        if max_prompt_len >= MAX_CONTEXT_LENGTH:
            print(f"WARNING: Truncation occurring ({max_prompt_len - MAX_CONTEXT_LENGTH} tokens over limit)")
        else:
            print(f"No truncation ({MAX_CONTEXT_LENGTH - max_prompt_len} tokens headroom)")
        print(f"\nDelta log-probability statistics:")
        print(f"Mean delta: {mean_delta:.6f}")
        print(f"Std delta: {std_delta:.6f}")
        print(f"2-sigma threshold: {threshold:.6f}")
        
        selected_deltas = [p['delta_log_prob'] for r in all_results for p in r['passage_analysis'] if p['is_selected'] == 1]
        non_selected_deltas = [p['delta_log_prob'] for r in all_results for p in r['passage_analysis'] if p['is_selected'] == 0]
        selected_norm_deltas = [p['normalized_delta'] for r in all_results for p in r['passage_analysis'] if p['is_selected'] == 1]
        non_selected_norm_deltas = [p['normalized_delta'] for r in all_results for p in r['passage_analysis'] if p['is_selected'] == 0]
        
        misattributed_count = 0
        for r in all_results:
            max_delta_passage = max(r['passage_analysis'], key=lambda x: x['normalized_delta'])
            if max_delta_passage['is_selected'] == 0:
                misattributed_count += 1
        
        if selected_deltas and non_selected_deltas:
            print(f"\nAttribution analysis (absolute delta):")
            print(f"Selected passages mean delta: {np.mean(selected_deltas):.6f}")
            print(f"Non-selected passages mean delta: {np.mean(non_selected_deltas):.6f}")
            print(f"Difference: {np.mean(selected_deltas) - np.mean(non_selected_deltas):.6f}")
            if np.mean(selected_deltas) > np.mean(non_selected_deltas):
                print(f"Selected passages have higher attribution (expected)")
            else:
                print(f"Non-selected passages have higher attribution (unexpected)")
        
        if selected_norm_deltas and non_selected_norm_deltas:
            print(f"\nAttribution analysis (normalized delta):")
            print(f"Selected passages mean normalized delta: {np.mean(selected_norm_deltas):.6f}")
            print(f"Non-selected passages mean normalized delta: {np.mean(non_selected_norm_deltas):.6f}")
            print(f"Difference: {np.mean(selected_norm_deltas) - np.mean(non_selected_norm_deltas):.6f}")
            if np.mean(selected_norm_deltas) > np.mean(non_selected_norm_deltas):
                print(f"Selected passages have higher attribution (expected)")
            else:
                print(f"Non-selected passages have higher attribution (unexpected)")
        
        print(f"\nMisattribution analysis:")
        print(f"Samples where highest-delta passage is NOT selected: {misattributed_count}/{len(all_results)}")
        print(f"Misattribution rate: {100*misattributed_count/len(all_results):.2f}%")
        print(f"Correct attribution rate: {100*(1-misattributed_count/len(all_results)):.2f}%")

if __name__ == "__main__":
    main()
