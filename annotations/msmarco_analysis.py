import json
import random
from typing import List, Dict, Optional
import warnings
import gc

import numpy as np
import torch
from tqdm import tqdm
import spacy
from scipy.stats import norm

from utils import (
    setup_model_and_tokenizer,
    load_msmarco_dataset,
    format_msmarco_prompt,
    get_token_probability_from_input_ids,
    calculate_log_odds,
)

from config import (
    QUERY_TYPES, TOTAL_EXAMPLES, NULL_EXAMPLES,
    NUM_STOCHASTIC_REPEATS, BATCH_SIZE, USE_POSITIONAL_BINNING,
    Q_CRITICAL, Q_HARMFUL, PUNCTUATION_TOKENS,
    NULL_STATS_FILE, RESULTS_FILE, ATTRIBUTIONS_FILE, RANDOM_SEED,
    set_seed, validate_config,
    BOTTOM_K_PERCENT_FOR_NULL # Make sure to add this to your config
)

torch.backends.cuda.matmul.allow_tf32 = True

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    warnings.warn("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None


class TokenInfluenceAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        if USE_POSITIONAL_BINNING:
            self.null_distributions = {'first_10': [], 'middle_80': [], 'last_10': [], 'all_positions': []}
        else:
            self.null_distributions = {'all_positions': []}

    def get_positional_bin(self, token_idx: int, total_tokens: int) -> str:
        if not USE_POSITIONAL_BINNING:
            return 'all_positions'
        if total_tokens <= 1:
            return 'first_10'
        first_10_boundary = max(1, int(total_tokens * 0.1))
        last_10_boundary = int(total_tokens * 0.9)
        if token_idx < first_10_boundary:
            return 'first_10'
        elif token_idx >= last_10_boundary:
            return 'last_10'
        else:
            return 'middle_80'

    def get_token_attention_scores(self, prompt: str) -> List[Dict]:
        """
        Performs a single forward pass to get attention scores.
        Returns a list of dicts, each containing the token's index,
        score, and positional bin, EXCLUDING special tokens.
        """
        self.model.eval()
        
        input_ids_tensor = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)['input_ids']
        
        with torch.no_grad():
            outputs = self.model(input_ids_tensor, output_attentions=True)
        
        attentions = outputs.attentions 
        last_layer_attentions = attentions[-1]
        avg_head_attentions = last_layer_attentions.mean(dim=1).squeeze(0)
        scores_for_last_token = avg_head_attentions[-1, :].cpu().numpy()
        
        input_ids = input_ids_tensor.squeeze(0).tolist()
        special_ids = self.tokenizer.all_special_ids
        added_ids = list(self.tokenizer.added_tokens_decoder.keys())
        non_content_ids_set = set(special_ids + added_ids)
        
        results = []
        for idx, token_id in enumerate(input_ids):
            if token_id not in non_content_ids_set:
                results.append({
                    'token_idx': idx,
                    'score': scores_for_last_token[idx],
                    'positional_bin': self.get_positional_bin(idx, len(input_ids))
                })
        return results

    def calculate_attention_based_null_distribution(self, dataset, num_examples=None, bottom_k_percent=20):
        """
        [MODIFIED per request: Global Bottom-K, then Bin]
        Builds a null distribution by finding the bottom k% of scores globally,
        and then binning that subset.
        """
        if num_examples is None:
            num_examples = min(len(dataset), 150)

        for bin_name in self.null_distributions:
            self.null_distributions[bin_name] = []

        # Step 1: Collect ALL content token scores into a single global list.
        all_token_scores = []
        indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
        
        pbar = tqdm(indices, desc="Building Null (Step 1/2: Collecting Scores)")
        for i in pbar:
            row = dataset[i]
            prompt = format_msmarco_prompt(row, self.tokenizer)
            try:
                # scores_data is a list of dicts: [{'token_idx': ..., 'score': ..., 'positional_bin': ...}]
                scores_data = self.get_token_attention_scores(prompt)
                all_token_scores.extend(scores_data)
            except Exception as e:
                print(f"Warning: Error getting attention for example {i}: {e}")
                continue
        
        print(f"Collected {len(all_token_scores)} total content token scores.")

        # Step 2: Identify the bottom-k tokens from the GLOBAL pool. ("Importance" first)
        all_token_scores.sort(key=lambda x: abs(x['score']))
        num_to_keep = int(len(all_token_scores) * (bottom_k_percent / 100.0))
        bottom_k_tokens = all_token_scores[:num_to_keep]
        
        print(f"Using the {len(bottom_k_tokens)} lowest-attention tokens for the null distribution.")
        
        # Step 3: Bin the results AFTER identifying the bottom-k. ("Bin" second)
        for token_data in bottom_k_tokens:
            bin_name = token_data['positional_bin']
            score = token_data['score']
            if bin_name in self.null_distributions:
                self.null_distributions[bin_name].append(score)
            # We still keep an 'all_positions' for fallback.
            self.null_distributions['all_positions'].append(score)
            
        print("Finished building null distributions.")
        
        # Step 4: Calculate final statistics for each bin
        null_stats = {}
        for bin_name, values in self.null_distributions.items():
            if values:
                null_stats[bin_name] = {
                    'mean': np.mean(values), 'std': np.std(values), 'count': len(values),
                    'percentiles': {'95': np.percentile(values, 95), '99': np.percentile(values, 99)}
                }
            else:
                null_stats[bin_name] = {
                    'mean': 0.0, 'std': 0.01, 'count': 0,
                    'percentiles': {'95': 0.02, '99': 0.03}
                }
        
        return null_stats

    def analyze_token_attention(self, prompt: str) -> Dict:
        """
        Gets token importance scores using the fast attention-based method.
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        tokens = [self.tokenizer.decode([t]) for t in input_ids]
        
        attention_results = self.get_token_attention_scores(prompt)
        
        token_results = []
        for res in attention_results:
            idx = res['token_idx']
            token_results.append({
                'token_idx': idx,
                'token': tokens[idx],
                'mean_influence': res['score'], # The attention 'score' is now the influence metric
                'uncertainty': 0.0,
                'positional_bin': res['positional_bin']
            })
            
        token_results.sort(key=lambda x: abs(x['mean_influence']), reverse=True)
        
        baseline_log_odds = 0.0
        try:
            input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
                top_prob = torch.max(probs).item()
                baseline_log_odds = calculate_log_odds(top_prob)
        except Exception:
            pass

        return {
            'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
            'total_tokens': len(tokens),
            'analyzed_tokens': len(token_results),
            'baseline_log_odds': baseline_log_odds,
            'token_results': token_results
        }

    def calculate_shrunken_z_scores(self, token_results: List[Dict], null_stats: Dict) -> List[Dict]:
        enhanced_results = []
        for token in token_results:
            bin_name = token['positional_bin']
            
            bin_stats = null_stats.get(bin_name)
            if not bin_stats or bin_stats['count'] == 0:
                bin_stats = null_stats.get('all_positions', {'mean': 0.0, 'std': 0.01})

            null_mean = bin_stats['mean']
            null_std = max(bin_stats['std'], 1e-8)
            
            shrunken_z_score = (token['mean_influence'] - null_mean) / null_std
            
            enhanced_token = token.copy()
            enhanced_token['shrunken_z_score'] = shrunken_z_score
            enhanced_results.append(enhanced_token)
        return enhanced_results

    def extract_linguistic_spans(self, prompt: str, token_results: List[Dict]) -> List[Dict]:
        if nlp is None:
            spans = []
            for token in token_results:
                spans.append({
                    'start_idx': token['token_idx'],
                    'end_idx': token['token_idx'] + 1,
                    'text': token['token'], 'tokens': [token],
                    'max_abs_z_score': abs(token['shrunken_z_score']),
                    'representative_z_score': token['shrunken_z_score'], 'span_type': 'token'
                })
            return spans
        
        doc = nlp(prompt)
        token_map = {res['token_idx']: res for res in token_results}
        spans = []
        for idx in sorted(token_map.keys()):
            token_result = token_map[idx]
            spans.append({
                'start_idx': idx, 'end_idx': idx + 1, 'text': token_result['token'],
                'tokens': [token_result], 'max_abs_z_score': abs(token_result['shrunken_z_score']),
                'representative_z_score': token_result['shrunken_z_score'], 'span_type': 'token'
            })
        return sorted(spans, key=lambda x: x['start_idx'])

    def benjamini_hochberg_procedure(self, p_values: List[float], q_value: float) -> List[bool]:
        if not p_values: return []
        n = len(p_values)
        indexed_p_values = sorted([(p, i) for i, p in enumerate(p_values)])
        significant = [False] * n
        for k in range(n - 1, -1, -1):
            p_val, original_idx = indexed_p_values[k]
            threshold = (k + 1) / n * q_value
            if p_val <= threshold:
                for j in range(k + 1):
                    _, idx = indexed_p_values[j]
                    significant[idx] = True
                break
        return significant

    def apply_fdr_control(self, spans: List[Dict]) -> List[Dict]:
        p_positive = [1 - norm.cdf(s['representative_z_score']) for s in spans]
        p_negative = [norm.cdf(s['representative_z_score']) for s in spans]
        significant_critical = self.benjamini_hochberg_procedure(p_positive, Q_CRITICAL)
        significant_harmful = self.benjamini_hochberg_procedure(p_negative, Q_HARMFUL)
        for i, span in enumerate(spans):
            z_score = span['representative_z_score']
            if significant_critical[i] and z_score > 0:
                span['label'], span['significance_type'] = 'TC', 'critical'
            elif significant_harmful[i] and z_score < 0:
                span['label'], span['significance_type'] = 'TH', 'harmful'
            else:
                span['label'], span['significance_type'] = 'TR', 'not_significant'
            span['p_positive'], span['p_negative'] = p_positive[i], p_negative[i]
        return spans

    def propagate_labels_to_tokens(self, token_results: List[Dict], labeled_spans: List[Dict]) -> List[Dict]:
        final_tokens = sorted(token_results, key=lambda x: x['token_idx'])
        token_map = {t['token_idx']: t for t in final_tokens}
        for token in final_tokens:
            token['final_label'] = 'TR'
            token['span_info'] = None
        for span in labeled_spans:
            for token_in_span in span['tokens']:
                idx = token_in_span['token_idx']
                if idx in token_map:
                    token_map[idx]['final_label'] = span['label']
                    token_map[idx]['span_info'] = {
                        'span_text': span['text'], 'span_type': span['span_type'],
                        'significance_type': span['significance_type'],
                        'representative_z_score': span['representative_z_score']
                    }
        return list(token_map.values())

    def complete_analysis_pipeline(self, prompt: str, null_stats: Dict, pbar=None) -> Dict:
        if pbar: pbar.set_postfix({"phase": "1/3", "step": "attention"})
        
        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Empty prompt provided")
            
        phase1_results = self.analyze_token_attention(prompt)
        
        if pbar: pbar.set_postfix({"phase": "2/3", "step": "Z-scores"})
        tokens_with_z_scores = self.calculate_shrunken_z_scores(
            phase1_results['token_results'], null_stats
        )
        
        if pbar: pbar.set_postfix({"phase": "3/3", "step": "spans"})
        linguistic_spans = self.extract_linguistic_spans(prompt, tokens_with_z_scores)
        
        if pbar: pbar.set_postfix({"phase": "3/3", "step": "FDR"})
        labeled_spans = self.apply_fdr_control(linguistic_spans)
        
        if pbar: pbar.set_postfix({"phase": "3/3", "step": "labels"})
        final_tokens = self.propagate_labels_to_tokens(tokens_with_z_scores, labeled_spans)
        
        if not final_tokens:
            print(f"Warning: No tokens in final_tokens for prompt: {prompt[:50]}...")
        
        return {
            'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
            'total_tokens': phase1_results['total_tokens'],
            'baseline_log_odds': phase1_results['baseline_log_odds'],
            'tokens': final_tokens,
            'spans': labeled_spans,
            'summary': {
                'critical_tokens': sum(1 for t in final_tokens if t.get('final_label') == 'TC'),
                'harmful_tokens': sum(1 for t in final_tokens if t.get('final_label') == 'TH'),
                'redundant_tokens': sum(1 for t in final_tokens if t.get('final_label') == 'TR'),
                'critical_spans': sum(1 for s in labeled_spans if s.get('label') == 'TC'),
                'harmful_spans': sum(1 for s in labeled_spans if s.get('label') == 'TH'),
                'redundant_spans': sum(1 for s in labeled_spans if s.get('label') == 'TR')
            }
        }

def analyze_msmarco_dataset():
    set_seed(RANDOM_SEED)
    validate_config()
    
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    
    print("Loading MS MARCO dataset...")
    dataset = load_msmarco_dataset(
        query_types=QUERY_TYPES,
        total_examples=TOTAL_EXAMPLES,
        seed=RANDOM_SEED
    )
    
    analyzer = TokenInfluenceAnalyzer(model, tokenizer)
    
    # --- CHANGE #1: Call the new attention-based null distribution builder ---
    print("\n=== Building Attention-Based Null Distribution from Bottom-K Least Influential Tokens ===")
    null_stats = analyzer.calculate_attention_based_null_distribution(
        dataset, 
        num_examples=NULL_EXAMPLES,
        bottom_k_percent=BOTTOM_K_PERCENT_FOR_NULL
    )
    
    print("\n=== Null Distribution Statistics ===")
    for bin_name, stats in null_stats.items():
        if stats['count'] > 0:
            print(f"\n{bin_name}:")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std: {stats['std']:.6f}")
            print(f"  95th percentile: {stats['percentiles']['95']:.6f}")
            print(f"  99th percentile: {stats['percentiles']['99']:.6f}")
            print(f"  Sample size: {stats['count']}")
    
    with open(NULL_STATS_FILE, 'w') as f:
        json.dump(null_stats, f, indent=2)
    print(f"\nNull distribution statistics saved to {NULL_STATS_FILE}")
    
    print(f"\n=== Analyzing {len(dataset)} MS MARCO Examples with Attention-Based Pipeline ===")
    
    all_results = []
    # Use a more direct way to iterate over the dataset
    pbar = tqdm(dataset, desc="Analyzing examples", mininterval=0.1, dynamic_ncols=True)
    
    for i, row in enumerate(pbar):
        try:
            prompt = format_msmarco_prompt(row, tokenizer)
            
            # --- CHANGE #2: This call is correct and now uses the attention-based null_stats ---
            results = analyzer.complete_analysis_pipeline(prompt, null_stats, pbar)
            
            results['query'] = row.get('query', '')[:100]
            all_results.append(results)
            
            pbar.set_postfix({
                "critical": results['summary']['critical_tokens'],
                "harmful": results['summary']['harmful_tokens']
            })
            
            # Optional: Clean up memory aggressively inside the loop
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            pbar.set_postfix({"error": str(e)[:30]})
            print(f"\nWarning: Error processing example {i}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue
            
    # --- (The rest of the function for saving results can remain the same) ---
    with open(RESULTS_FILE, 'w') as f:
        # The `all_results` structure is complex but should be JSON-serializable
        json.dump(all_results, f, indent=2)

    with open(ATTRIBUTIONS_FILE, 'w') as f:
        detailed_attributions = []
        for i, result in enumerate(all_results):
            for token in result['tokens']:
                detailed_attributions.append({
                    'example_id': i,
                    'query': result.get('query', ''),
                    'token': token['token'],
                    'token_idx': token['token_idx'],
                    'attention_score': token['mean_influence'], # Renaming for clarity
                    'shrunken_z_score': token.get('shrunken_z_score', 0),
                    'final_label': token.get('final_label'),
                    'positional_bin': token['positional_bin'],
                    'span_info': token.get('span_info')
                })
        json.dump(detailed_attributions, f, indent=2)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"Detailed token attributions saved to {ATTRIBUTIONS_FILE}")
    
    # ... (Final summary printouts)
     
if __name__ == "__main__":
    analyze_msmarco_dataset()