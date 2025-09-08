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
    set_seed, validate_config
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
            self.null_distributions = {'first_10': [], 'middle_80': [], 'last_10': []}
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
    
    def is_punctuation_token(self, token_id: int) -> bool:
        decoded = self.tokenizer.decode([token_id], skip_special_tokens=False).strip()
        return any(char in PUNCTUATION_TOKENS for char in decoded)
    
    def analyze_token_influence(
        self,
        prompt: str,
        target_tokens: Optional[List[int]] = None,
        num_repeats: int = NUM_STOCHASTIC_REPEATS,
        show_progress: bool = True,
        external_pbar=None,
        batch_size: int = BATCH_SIZE  # Process tokens in batches for better GPU utilization
    ) -> Dict:
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        tokens = [self.tokenizer.decode([t]) for t in input_ids]


        # --- START: MODIFIED LOGIC TO EXCLUDE SPECIAL TOKENS FROM ANALYSIS ---

        # Step 1: Create a definitive 'whitelist' of all valid content token indices.
        special_ids_set = set(self.tokenizer.all_special_ids)

        # Get the "added" tokens (e.g., <|user|>, <|assistant|>)
        # We get their IDs by taking the keys from the added_tokens_decoder dictionary.
        added_ids = set(self.tokenizer.added_tokens_decoder.keys())

        # Combine both lists and create a single set for efficient lookup.
        # This is the complete set of all tokens to be excluded from analysis.
        non_content_ids_set = special_ids_set.union(added_ids)

        content_token_indices = [
            idx for idx, token_id in enumerate(input_ids) 
            if token_id not in non_content_ids_set
        ]

        if target_tokens is None:
            target_tokens = content_token_indices #list(range(len(tokens)))
        else:
            content_indices_set = set(content_token_indices)
            target_tokens = [t for t in target_tokens if t < len(tokens) and t in content_indices_set]
        
        baseline_input = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        baseline_input = baseline_input.to(self.model.device)
        
        with torch.no_grad():
            baseline_outputs = self.model(baseline_input)
            baseline_logits = baseline_outputs.logits[0, -1, :]
            baseline_probs = torch.softmax(baseline_logits, dim=-1)
            # Get top prediction
            top_token_id = torch.argmax(baseline_probs).item()
            baseline_prob = baseline_probs[top_token_id].item()
            baseline_log_odds = calculate_log_odds(baseline_prob)
        
        # Clean up
        del baseline_outputs, baseline_logits
        torch.cuda.empty_cache()
        
        # Analyze influence of each target token
        token_results = []
        
        # Enable dropout for stochastic analysis
        self.model.train()
        
        # Sequential processing for memory efficiency
        def process_token_batch(batch_indices):
            # Pre-allocate results for this batch
            batch_results = []
            
            # Create all masked inputs for the batch at once
            batch_masked_inputs = []
            valid_indices = []
            
            for idx in batch_indices:
                if idx >= len(input_ids):
                    continue
                    
                # Create masked input (remove token at idx)
                masked_ids = input_ids[:idx] + input_ids[idx+1:]
                if len(masked_ids) == 0:
                    continue
                    
                batch_masked_inputs.append(masked_ids)
                valid_indices.append(idx)
            
            if not batch_masked_inputs:
                return
            
            # Process all valid tokens in this batch
            for i, (masked_ids, idx) in enumerate(zip(batch_masked_inputs, valid_indices)):
                masked_input = torch.tensor(masked_ids, dtype=torch.long).unsqueeze(0)
                masked_input = masked_input.to(self.model.device)
                
                influences = []
                # Vectorize repeated computations where possible
                for _ in range(num_repeats):
                    try:
                        with torch.no_grad():
                            masked_prob = get_token_probability_from_input_ids(
                                self.model, self.tokenizer, masked_input, "A"
                            )
                            masked_log_odds = calculate_log_odds(masked_prob)
                            influence = baseline_log_odds - masked_log_odds
                            influences.append(influence)
                            
                            # Clear intermediate tensors immediately
                            del masked_prob
                            
                    except Exception as e:
                        print(f"Error processing token {idx}: {e}")
                        continue
                
                # Clean up masked_input
                del masked_input
                
                if influences:
                    mean_influence = np.mean(influences)
                    std_influence = np.std(influences)
                    
                    token_results.append({
                        'token_idx': idx,
                        'token': tokens[idx],
                        'mean_influence': mean_influence,
                        'uncertainty': std_influence,
                        'is_punctuation': self.is_punctuation_token(input_ids[idx]),
                        'positional_bin': self.get_positional_bin(idx, len(tokens))
                    })
            
            # Force garbage collection after each batch
            torch.cuda.empty_cache()
            gc.collect()
        
        # Process tokens in batches for better performance
        for i in range(0, len(target_tokens), batch_size):
            batch_tokens = target_tokens[i:i+batch_size]
            batch_end = min(i + batch_size, len(target_tokens))
            
            if external_pbar:
                external_pbar.set_postfix({"phase": "1/3", "tokens": f"{batch_end}/{len(target_tokens)}"})
            elif show_progress:
                progress_msg = f"Processing tokens {batch_end}/{len(target_tokens)} (influence analysis)"
                print(f"\r{progress_msg}", end="", flush=True)
            
            # Process batch with optimized memory usage
            process_token_batch(batch_tokens)
        
        if show_progress and not external_pbar:
            print()  # New line after progress
        
        # Disable dropout
        self.model.eval()
        
        # Sort by influence magnitude
        token_results.sort(key=lambda x: abs(x['mean_influence']), reverse=True)
        
        return {
            'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
            'total_tokens': len(tokens),
            'analyzed_tokens': len(target_tokens),
            'baseline_log_odds': baseline_log_odds,
            'token_results': token_results
        }
    
    def calculate_empirical_null_distribution(self, dataset, num_examples=None):
        if num_examples is None:
            num_examples = min(len(dataset), 50)
        
        if USE_POSITIONAL_BINNING:
            self.null_distributions = {'first_10': [], 'middle_80': [], 'last_10': []}
        else:
            self.null_distributions = {'all_positions': []}
        
        # Random sampling for null distribution
        indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
        
        # Pre-count punctuation tokens for progress display
        total_punct_tokens = 0
        for i in indices:
            row = dataset[i]
            prompt = format_msmarco_prompt(row, self.tokenizer)
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            for token_id in input_ids:
                if self.is_punctuation_token(token_id):
                    total_punct_tokens += 1
        
        processed_punct_tokens = 0
        pbar = tqdm(indices, desc="Processing examples", mininterval=0.1)
        for i in pbar:
            try:
                row = dataset[i]
                prompt = format_msmarco_prompt(row, self.tokenizer)
                
                # Tokenize to find punctuation
                input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                
                # Find punctuation token indices
                punctuation_indices = []
                for idx, token_id in enumerate(input_ids):
                    if self.is_punctuation_token(token_id):
                        punctuation_indices.append(idx)
                
                if not punctuation_indices:
                    pbar.set_postfix({"status": "no punctuation"})
                    continue
                
                # Analyze only punctuation tokens with progress update
                pbar.set_postfix({"status": f"analyzing {len(punctuation_indices)} punct tokens"})
                results = self.analyze_token_influence(
                    prompt, 
                    target_tokens=punctuation_indices,
                    num_repeats=3,
                    show_progress=False,
                    batch_size=BATCH_SIZE
                )
                
                # Add to appropriate bins and track progress
                for token_result in results['token_results']:
                    if token_result['is_punctuation']:
                        processed_punct_tokens += 1
                        if USE_POSITIONAL_BINNING:
                            bin_name = token_result['positional_bin']
                            if bin_name in self.null_distributions:
                                self.null_distributions[bin_name].append(token_result['mean_influence'])
                        else:
                            self.null_distributions['all_positions'].append(token_result['mean_influence'])
                
                pbar.set_postfix({"punct_tokens": f"{processed_punct_tokens}/{total_punct_tokens}"})
                
                # Clear GPU cache after each example
                torch.cuda.empty_cache()
                
            except Exception as e:
                pbar.set_postfix({"error": str(e)[:20]})
                print(f"Warning: Error processing example {i}: {e}")
                continue
        
        # Calculate statistics for each bin
        null_stats = {}
        for bin_name, values in self.null_distributions.items():
            if values:
                null_stats[bin_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values),
                    'percentiles': {
                        '95': np.percentile(values, 95),
                        '99': np.percentile(values, 99)
                    }
                }
            else:
                null_stats[bin_name] = {
                    'mean': 0.0,
                    'std': 0.01,  # Default small value
                    'count': 0,
                    'percentiles': {'95': 0.02, '99': 0.03}
                }
        
        return null_stats
    
    def calculate_shrunken_z_scores(self, token_results: List[Dict], null_stats: Dict) -> List[Dict]:
        enhanced_results = []
        
        for token in token_results:
            bin_name = token['positional_bin']
            if bin_name not in null_stats:
                null_mean, null_std = 0.0, 0.01
            else:
                null_mean = null_stats[bin_name]['mean']
                null_std = max(null_stats[bin_name]['std'], 1e-8)
            token_uncertainty = max(token.get('uncertainty', 0.01), 1e-8)
            shrinkage = min(0.8, max(0.2, token_uncertainty / (null_std + 1e-8)))
            shrunken_mean = shrinkage * token['mean_influence'] + (1 - shrinkage) * null_mean
            shrunken_z_score = (shrunken_mean - null_mean) / (null_std + 1e-8)
            
            enhanced_token = token.copy()
            enhanced_token['shrunken_z_score'] = shrunken_z_score
            enhanced_results.append(enhanced_token)
        
        return enhanced_results
    
    def extract_linguistic_spans(self, prompt: str, token_results: List[Dict]) -> List[Dict]:
        if nlp is None:
            # Fallback: treat each token as its own span
            spans = []
            for i, token in enumerate(token_results):
                spans.append({
                    'start_idx': i,
                    'end_idx': i + 1,
                    'text': token['token'],
                    'tokens': [token],
                    'max_abs_z_score': abs(token['shrunken_z_score']),
                    'representative_z_score': token['shrunken_z_score'],
                    'span_type': 'token'
                })
            return spans
        
        # Parse with spaCy
        doc = nlp(prompt)
        
        # Create token index mapping
        token_map = {}
        for i, token_result in enumerate(token_results):
            token_map[i] = token_result
        
        # Extract spans: noun phrases, named entities, and individual tokens
        spans = []
        covered_tokens = set()
        
        # Process noun phrases
        for chunk in doc.noun_chunks:
            start_char = chunk.start_char
            end_char = chunk.end_char
            
            # Find corresponding token indices using index-based mapping
            span_token_indices = []
            for i in range(len(token_results)):
                if i not in covered_tokens and len(span_token_indices) < len(chunk):
                    span_token_indices.append(i)
                    covered_tokens.add(i)
            
            if span_token_indices:
                span_tokens = [token_results[i] for i in span_token_indices]
                z_scores = [t['shrunken_z_score'] for t in span_tokens]
                max_abs_z = max([abs(z) for z in z_scores])
                representative_z = max(z_scores, key=abs)
                
                spans.append({
                    'start_idx': min(span_token_indices),
                    'end_idx': max(span_token_indices) + 1,
                    'text': chunk.text,
                    'tokens': span_tokens,
                    'max_abs_z_score': max_abs_z,
                    'representative_z_score': representative_z,
                    'span_type': 'noun_phrase'
                })
        
        # Process named entities
        for ent in doc.ents:
            span_token_indices = []
            for i in range(len(token_results)):
                if i not in covered_tokens and len(span_token_indices) < len(ent):
                    span_token_indices.append(i)
                    covered_tokens.add(i)
            
            if span_token_indices:
                span_tokens = [token_results[i] for i in span_token_indices]
                z_scores = [t['shrunken_z_score'] for t in span_tokens]
                max_abs_z = max([abs(z) for z in z_scores])
                representative_z = max(z_scores, key=abs)
                
                spans.append({
                    'start_idx': min(span_token_indices),
                    'end_idx': max(span_token_indices) + 1,
                    'text': ent.text,
                    'tokens': span_tokens,
                    'max_abs_z_score': max_abs_z,
                    'representative_z_score': representative_z,
                    'span_type': 'named_entity'
                })
        
        # Add remaining individual tokens
        for i, token_result in enumerate(token_results):
            if i not in covered_tokens:
                spans.append({
                    'start_idx': i,
                    'end_idx': i + 1,
                    'text': token_result['token'],
                    'tokens': [token_result],
                    'max_abs_z_score': abs(token_result['shrunken_z_score']),
                    'representative_z_score': token_result['shrunken_z_score'],
                    'span_type': 'token'
                })
        
        return sorted(spans, key=lambda x: x['start_idx'])
    
    def benjamini_hochberg_procedure(self, p_values: List[float], q_value: float) -> List[bool]:
        """
        Apply Benjamini-Hochberg FDR control procedure.
        
        Args:
            p_values: List of p-values
            q_value: Desired FDR level (e.g., 0.05)
        
        Returns:
            List of booleans indicating significance
        """
        if not p_values:
            return []
        
        n = len(p_values)
        # Sort p-values and keep track of original indices
        indexed_p_values = [(p, i) for i, p in enumerate(p_values)]
        indexed_p_values.sort()
        
        # Apply BH procedure
        significant = [False] * n
        for k in range(n - 1, -1, -1):
            p_val, original_idx = indexed_p_values[k]
            threshold = (k + 1) / n * q_value
            
            if p_val <= threshold:
                # Mark this and all smaller p-values as significant
                for j in range(k + 1):
                    _, idx = indexed_p_values[j]
                    significant[idx] = True
                break
        
        return significant
    
    def apply_fdr_control(self, spans: List[Dict]) -> List[Dict]:
        # Calculate p-values for each span
        p_positive = []  # For critical spans
        p_negative = []  # For harmful spans
        
        for span in spans:
            z_score = span['representative_z_score']
            
            p_pos = 1 - norm.cdf(z_score)  # P(Z >= z_score)
            p_neg = norm.cdf(z_score)      # P(Z <= z_score)
            
            p_positive.append(p_pos)
            p_negative.append(p_neg)
        
        significant_critical = self.benjamini_hochberg_procedure(p_positive, Q_CRITICAL)
        significant_harmful = self.benjamini_hochberg_procedure(p_negative, Q_HARMFUL)
        
        for i, span in enumerate(spans):
            z_score = span['representative_z_score']
            
            if significant_critical[i] and z_score > 0:
                span['label'] = 'TC'  # Critical
                span['significance_type'] = 'critical'
            elif significant_harmful[i] and z_score < 0:
                span['label'] = 'TH'  # Harmful
                span['significance_type'] = 'harmful'
            else:
                span['label'] = 'TR'  # Redundant
                span['significance_type'] = 'not_significant'
            
            # Add p-values for reference
            span['p_positive'] = p_positive[i]
            span['p_negative'] = p_negative[i]
        
        return spans
    
    def propagate_labels_to_tokens(self, token_results: List[Dict], labeled_spans: List[Dict]) -> List[Dict]:
        final_tokens = []
        for token in token_results:
            token_copy = token.copy()
            token_copy['final_label'] = 'TR'
            token_copy['span_info'] = None
            final_tokens.append(token_copy)
        
        for span in labeled_spans:
            start_idx = span['start_idx']
            end_idx = span['end_idx']
            
            for i in range(start_idx, end_idx):
                if i < len(final_tokens):
                    final_tokens[i]['final_label'] = span['label']
                    final_tokens[i]['span_info'] = {
                        'span_text': span['text'],
                        'span_type': span['span_type'],
                        'significance_type': span['significance_type'],
                        'representative_z_score': span['representative_z_score']
                    }
        
        return final_tokens
    
    def complete_analysis_pipeline(self, prompt: str, null_stats: Dict, pbar=None) -> Dict:
        if pbar:
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            pbar.set_postfix({"phase": "1/3", "tokens": f"0/{len(input_ids)}"})
        
        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Empty prompt provided")
            
        phase1_results = self.analyze_token_influence(prompt, show_progress=False, external_pbar=pbar)
        
        # Phase 3.1: Calculate shrunken Z-scores
        if pbar:
            pbar.set_postfix({"phase": "2/3", "step": "Z-scores"})
        tokens_with_z_scores = self.calculate_shrunken_z_scores(
            phase1_results['token_results'], null_stats
        )
        
        # Phase 3.2: Extract linguistic spans
        if pbar:
            pbar.set_postfix({"phase": "3/3", "step": "spans"})
        linguistic_spans = self.extract_linguistic_spans(prompt, tokens_with_z_scores)
        
        # Phase 3.3: Apply FDR control
        if pbar:
            pbar.set_postfix({"phase": "3/3", "step": "FDR"})
        labeled_spans = self.apply_fdr_control(linguistic_spans)
        
        # Phase 3.4: Project labels to tokens
        if pbar:
            pbar.set_postfix({"phase": "3/3", "step": "labels"})
        final_tokens = self.propagate_labels_to_tokens(tokens_with_z_scores, labeled_spans)
        
        # Validate results before compiling
        if not final_tokens:
            print(f"Warning: No tokens in final_tokens for prompt: {prompt[:50]}...")
        
        # Compile results
        results = {
            'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
            'total_tokens': len(final_tokens),
            'baseline_log_odds': phase1_results['baseline_log_odds'],
            'tokens': final_tokens,
            'spans': labeled_spans,
            'summary': {
                'critical_tokens': len([t for t in final_tokens if t.get('final_label') == 'TC']),
                'harmful_tokens': len([t for t in final_tokens if t.get('final_label') == 'TH']),
                'redundant_tokens': len([t for t in final_tokens if t.get('final_label') == 'TR']),
                'critical_spans': len([s for s in labeled_spans if s.get('label') == 'TC']),
                'harmful_spans': len([s for s in labeled_spans if s.get('label') == 'TH']),
                'redundant_spans': len([s for s in labeled_spans if s.get('label') == 'TR'])
            }
        }
        
        return results
    
    def calculate_significance(self, token_result: Dict, null_stats: Dict) -> Dict:
        bin_name = token_result['positional_bin']
        if bin_name not in null_stats:
            bin_stats = null_stats.get('all_positions', {'mean': 0.0, 'std': 0.01, 'percentiles': {'95': 0.02, '99': 0.03}})
        else:
            bin_stats = null_stats[bin_name]
        z_score = (token_result['mean_influence'] - bin_stats['mean']) / (bin_stats['std'] + 1e-8)
        is_significant_95 = abs(token_result['mean_influence']) > abs(bin_stats['percentiles']['95'])
        is_significant_99 = abs(token_result['mean_influence']) > abs(bin_stats['percentiles']['99'])
        
        return {
            'z_score': z_score,
            'is_significant_95': is_significant_95,
            'is_significant_99': is_significant_99,
            'null_mean': bin_stats['mean'],
            'null_std': bin_stats['std']
        }


def analyze_msmarco_dataset():
    # Set seed for reproducibility
    set_seed(RANDOM_SEED)
    
    # Validate configuration
    validate_config()
    
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    
    print("Loading MS MARCO dataset...")
    dataset = load_msmarco_dataset(
        query_types=QUERY_TYPES,
        total_examples=TOTAL_EXAMPLES,
        seed=RANDOM_SEED,
        only_selected_passages=True
    )
    
    # Initialize analyzer
    analyzer = TokenInfluenceAnalyzer(model, tokenizer)
    
    # Phase 2: Build null distribution first
    null_stats = analyzer.calculate_empirical_null_distribution(dataset, NULL_EXAMPLES)
    
    print("\n=== Null Distribution Statistics ===")
    for bin_name, stats in null_stats.items():
        print(f"\n{bin_name}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  95th percentile: {stats['percentiles']['95']:.6f}")
        print(f"  99th percentile: {stats['percentiles']['99']:.6f}")
        print(f"  Sample size: {stats['count']}")
    
    # Save null statistics
    with open(NULL_STATS_FILE, 'w') as f:
        json.dump(null_stats, f, indent=2)
    print(f"\nNull distribution statistics saved to {NULL_STATS_FILE}")
    
    # Phase 1-3: Complete analysis with statistical guarantees
    print(f"\n=== Analyzing {TOTAL_EXAMPLES} MS MARCO Examples with 3-Phase Pipeline ===")
    
    all_results = []
    pbar = tqdm(range(TOTAL_EXAMPLES), desc="Analyzing examples", mininterval=0.1, dynamic_ncols=True)
    
    for i in pbar:
        try:
            row = dataset[i]
            prompt = format_msmarco_prompt(row, tokenizer)
            
            # Run complete 3-phase analysis pipeline with memory management
            results = analyzer.complete_analysis_pipeline(prompt, null_stats, pbar)
            
            # Add query information
            results['query'] = row.get('query', '')[:100]
            
            all_results.append(results)
            
            # Update progress bar with stats
            pbar.set_postfix({
                "critical": results['summary']['critical_tokens'],
                "harmful": results['summary']['harmful_tokens']
            })
            
            # Force garbage collection and GPU cache clearing after each example
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            pbar.set_postfix({"error": str(e)[:30]})
            print(f"Warning: Error processing example {i}: {e}")
            
            # Clear GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue
        
        # Print summary for first few examples  
        if i < 3:
            print(f"\n--- Example {i+1} ---")
            print(f"Query: {results['query']}")
            print(f"Tokens: {results['total_tokens']} | Critical: {results['summary']['critical_tokens']} | Harmful: {results['summary']['harmful_tokens']}")
    
    # Save comprehensive results
    with open(RESULTS_FILE, 'w') as f:
        # Convert to serializable format
        serializable_results = []
        for result in all_results:
            serializable_result = {
                'query': result['query'],
                'total_tokens': result['total_tokens'],
                'baseline_log_odds': float(result['baseline_log_odds']),
                'summary': result['summary'],
                'critical_spans': [
                    {
                        'text': s['text'],
                        'span_type': s['span_type'],
                        'z_score': float(s['representative_z_score']),
                        'p_value': float(s['p_positive'])
                    }
                    for s in result['spans'] if s['label'] == 'TC'
                ],
                'harmful_spans': [
                    {
                        'text': s['text'],
                        'span_type': s['span_type'],
                        'z_score': float(s['representative_z_score']),
                        'p_value': float(s['p_negative'])
                    }
                    for s in result['spans'] if s['label'] == 'TH'
                ],
                'token_labels': [
                    {
                        'token': t['token'],
                        'attribution': float(t['mean_influence']),
                        'std_deviation': float(t['uncertainty']),
                        'class': {'TC': 'critical', 'TH': 'harmful', 'TR': 'redundant'}.get(t['final_label'], t['final_label']),
                        'shrunken_z_score': float(t.get('shrunken_z_score', 0)),
                        'positional_bin': t['positional_bin'],
                        'token_idx': t['token_idx']
                    }
                    for t in result['tokens']
                ]
            }
            serializable_results.append(serializable_result)
        
        json.dump(serializable_results, f, indent=2)
    
    # Save detailed token attributions
    detailed_attributions = []
    for i, result in enumerate(all_results):
        for token in result['tokens']:
            detailed_attributions.append({
                'example_id': i,
                'query': result.get('query', '')[:100],
                'token': token['token'],
                'attribution': float(token['mean_influence']),
                'std_deviation': float(token['uncertainty']),
                'class': {'TC': 'critical', 'TH': 'harmful', 'TR': 'redundant'}.get(token['final_label'], token['final_label']),
                'shrunken_z_score': float(token.get('shrunken_z_score', 0)),
                'positional_bin': token['positional_bin'],
                'token_idx': token['token_idx'],
                'span_info': token.get('span_info')
            })
    
    with open(ATTRIBUTIONS_FILE, 'w') as f:
        json.dump(detailed_attributions, f, indent=2)
    
    print(f"\n=== Phase 3 Analysis Complete ===")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"Detailed token attributions saved to {ATTRIBUTIONS_FILE}")
    print(f"Analyzed {len(all_results)} examples")
    
    # Comprehensive summary statistics
    total_critical = sum(r['summary']['critical_tokens'] for r in all_results)
    total_harmful = sum(r['summary']['harmful_tokens'] for r in all_results)
    total_redundant = sum(r['summary']['redundant_tokens'] for r in all_results)
    total_tokens = sum(r['total_tokens'] for r in all_results)
    
    total_critical_spans = sum(r['summary']['critical_spans'] for r in all_results)
    total_harmful_spans = sum(r['summary']['harmful_spans'] for r in all_results)
    
    print(f"\nFinal Summary (All Examples):")
    print(f"  Total tokens analyzed: {total_tokens}")
    print(f"  Critical tokens: {total_critical} ({100*total_critical/total_tokens:.1f}%)")
    print(f"  Harmful tokens: {total_harmful} ({100*total_harmful/total_tokens:.1f}%)")
    print(f"  Redundant tokens: {total_redundant} ({100*total_redundant/total_tokens:.1f}%)")
    print(f"  Critical spans: {total_critical_spans}")
    print(f"  Harmful spans: {total_harmful_spans}")
    
    return all_results


if __name__ == "__main__":
    # Run analysis with config parameters
    analyze_msmarco_dataset()
