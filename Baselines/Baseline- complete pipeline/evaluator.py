"""
Evaluation Pipeline - Phase 3
Handles evaluation using multiple metrics including MS-MARCO, BLEU, ROUGE, and LLM judge
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

from utils import BaseComponent, ScaleDownAPI
from metrics import MSMARCOEvaluator, BLEUEvaluator, ROUGEEvaluator, LLMJudgeEvaluator


logger = logging.getLogger(__name__)


class EvaluationPipeline(BaseComponent):
    """Handles evaluation phase with multiple metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize API client for LLM judge
        api_config = self.config['api_config']
        self.api_client = ScaleDownAPI(
            api_key=api_config['api_key'],
            base_url=api_config['base_url'],
            model=api_config['model']
        )
        
        # Initialize evaluator registry lazily to avoid requiring optional deps
        # unless explicitly requested
        self.evaluators: Dict[str, Any] = {
            'llm_judge': LLMJudgeEvaluator(self.api_client)
        }
        
        # Default metrics to use
        eval_config = self.config.get('evaluation', {})
        self.default_metrics = [eval_config.get('metric', 'llm_judge')]

    def _get_or_create_evaluator(self, metric: str):
        """Return an evaluator instance for the given metric, creating it lazily.
        Raises ImportError if the required dependency is missing.
        """
        if metric in self.evaluators:
            return self.evaluators[metric]
        
        if metric == 'bleu':
            evaluator = BLEUEvaluator()
        elif metric == 'rouge':
            evaluator = ROUGEEvaluator()
        elif metric == 'msmarco':
            evaluator = MSMARCOEvaluator()
        elif metric == 'llm_judge':
            evaluator = self.evaluators['llm_judge']
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        self.evaluators[metric] = evaluator
        return evaluator
        
    def get_response_types(self, example: Dict) -> List[str]:
        """Auto-detect response types from example"""
        response_types = []
        metadata_keys = {'query_id', 'query', 'ground_truth', 'contexts', 'is_selected'}
        
        for key in example.keys():
            if key not in metadata_keys and isinstance(example[key], dict):
                if 'response' in example[key]:
                    response_types.append(key)
        
        return response_types
    
    def has_evaluations(self, example: Dict, metric: str) -> Dict[str, bool]:
        """Check which response types already have evaluations for given metric"""
        eval_status = {}
        response_types = self.get_response_types(example)
        
        for response_type in response_types:
            if ('evaluations' in example[response_type] and 
                metric in example[response_type]['evaluations']):
                eval_status[response_type] = True
            else:
                eval_status[response_type] = False
        
        return eval_status
    
    def evaluate_example(self, example: Dict, metrics: Optional[List[str]] = None) -> Dict:
        """Evaluate a single example with specified metrics"""
        if metrics is None:
            metrics = self.default_metrics
        
        ground_truth = example.get('ground_truth', '')
        response_types = self.get_response_types(example)
        
        for response_type in response_types:
            if 'response' not in example[response_type]:
                continue
                
            response = example[response_type]['response']
            if not response:
                continue
            
            # Initialize evaluations dict if not exists
            if 'evaluations' not in example[response_type]:
                example[response_type]['evaluations'] = {}
            
            # Evaluate with each metric
            for metric in metrics:
                # Skip if evaluation already exists
                eval_status = self.has_evaluations(example, metric)
                if eval_status.get(response_type, False):
                    logger.info(f"  Evaluation for {response_type} with {metric} already exists, skipping")
                    continue
                
                logger.info(f"  Evaluating {response_type} with {metric}...")
                
                try:
                    evaluator = self._get_or_create_evaluator(metric)
                    
                    # Get metric-specific parameters
                    eval_config = self.config.get('evaluation', {})
                    params = eval_config.get('parameters', {})
                    
                    if metric == 'llm_judge':
                        judge_model = params.get('judge_model')
                        score = evaluator.evaluate(ground_truth, response, judge_model=judge_model)
                    else:
                        score = evaluator.evaluate(ground_truth, response)
                    
                    example[response_type]['evaluations'][metric] = score
                    
                except ImportError as e:
                    # Optional dependency not available for this metric
                    logger.warning(f"{metric.upper()} evaluator unavailable: {e}")
                    example[response_type]['evaluations'][metric] = None
                    continue
                except Exception as e:
                    logger.error(f"Error evaluating {response_type} with {metric}: {e}")
                    example[response_type]['evaluations'][metric] = None
        
        return example
    
    def evaluate_from_file(self, input_file: str, metrics: Optional[List[str]] = None) -> List[Dict]:
        """Evaluate responses from input file"""
        logger.info(f"Loading data from {input_file}")
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        results = []
        for i, example in enumerate(data):
            logger.info(f"Evaluating example {i+1}/{len(data)}: {example.get('query', '')[:50]}...")
            
            try:
                result = self.evaluate_example(example, metrics)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating example {i+1}: {e}")
                results.append(example)  # Keep original even if evaluation failed
                continue
        
        logger.info(f"Evaluation completed. Processed {len(results)} examples")
        return results
    
    def export_msmarco_format(self, results: List[Dict], output_dir: Path):
        """Export results in MS-MARCO evaluation format"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        response_types = self.get_response_types(results[0]) if results else []
        
        for response_type in response_types:
            # Create reference and candidate files for each response type
            ref_file = output_dir / f"{response_type}_ref.txt"
            cand_file = output_dir / f"{response_type}_cand.txt"
            
            with open(ref_file, 'w') as ref_f, open(cand_file, 'w') as cand_f:
                for example in results:
                    ground_truth = example.get('ground_truth', '')
                    response = example.get(response_type, {}).get('response', '')
                    
                    ref_f.write(ground_truth + '\n')
                    cand_f.write(response + '\n')
            
            logger.info(f"MS-MARCO files for {response_type} saved to {output_dir}")
    
    def print_evaluation_table(self, results: List[Dict], metrics: Optional[List[str]] = None,
                               csv_path: Optional[Path] = None):
        """Print comprehensive evaluation table and optionally save to CSV.
        
        Args:
            results: Aggregated evaluation results
            metrics: Optional list of metrics to include
            csv_path: Optional path to save the table as CSV
        """
        if not results:
            logger.warning("No results to display")
            return
        
        # Auto-detect metrics if not specified
        if metrics is None:
            metrics = set()
            for example in results:
                response_types = self.get_response_types(example)
                for response_type in response_types:
                    if 'evaluations' in example[response_type]:
                        metrics.update(example[response_type]['evaluations'].keys())
            metrics = sorted(list(metrics))
        
        response_types = self.get_response_types(results[0])
        
        # Collect statistics
        stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        def _parse_rate_to_percent(value: Any) -> Optional[float]:
            """Parse compression rate into percent (float). Accepts formats like
            30, 0.3, "30%", "0.3", returns 30.0 for all equivalents."""
            try:
                if value is None:
                    return None
                if isinstance(value, (int, float)):
                    # Heuristic: values <= 1.0 are likely ratios; convert to percent
                    return float(value) * 100.0 if float(value) <= 1.0 else float(value)
                s = str(value).strip()
                if s.endswith('%'):
                    return float(s[:-1])
                # If looks like a ratio (<=1), treat as percent
                v = float(s)
                return v * 100.0 if v <= 1.0 else v
            except Exception:
                return None
        
        def _parse_ratio_to_x(value: Any) -> Optional[float]:
            """Parse compression ratio into x-multiplier (float). Accepts formats like
            3.3, "3.3x", "1.0x" and returns 3.3, 1.0, etc."""
            try:
                if value is None:
                    return None
                if isinstance(value, (int, float)):
                    return float(value)
                s = str(value).strip().lower()
                if s.endswith('x'):
                    s = s[:-1]
                return float(s)
            except Exception:
                return None
        
        for example in results:
            for response_type in response_types:
                if response_type not in example:
                    continue
                    
                evaluations = example[response_type].get('evaluations', {})
                for metric in metrics:
                    if metric in evaluations and evaluations[metric] is not None:
                        stats[response_type][metric]['scores'].append(evaluations[metric])
                
                # Track compression stats if available (non-original only)
                if response_type != 'original':
                    comp_result = example[response_type].get('compression_result', {})
                    if comp_result:
                        rate_percent = _parse_rate_to_percent(comp_result.get('compression_rate'))
                        ratio_x = _parse_ratio_to_x(comp_result.get('compression_ratio'))
                        if rate_percent is not None:
                            stats[response_type]['compression']['rate_percent'].append(rate_percent)
                        if ratio_x is not None:
                            stats[response_type]['compression']['ratio_x'].append(ratio_x)
        
        # Create summary table
        table_data = []
        for response_type in response_types:
            row = {'Response Type': response_type.replace('_', ' ').title()}
            
            for metric in metrics:
                scores = stats[response_type][metric]['scores']
                if scores:
                    if metric == 'llm_judge':
                        # Binary metric
                        matches = sum(scores)
                        total = len(scores)
                        accuracy = (matches / total * 100) if total > 0 else 0
                        row[f'{metric.upper()} (%)'] = f"{accuracy:.1f}"
                        row[f'{metric.upper()} (n)'] = f"{matches}/{total}"
                    else:
                        # Continuous metrics (BLEU, ROUGE)
                        avg_score = sum(scores) / len(scores)
                        row[f'{metric.upper()}'] = f"{avg_score:.3f}"
                else:
                    row[f'{metric.upper()}'] = 'N/A'
            
            # Add token information if available
            token_counts = []
            for example in results:
                if response_type in example:
                    if response_type == 'original':
                        token_count = example[response_type].get('token_count', 0)
                    else:
                        comp_result = example[response_type].get('compression_result', {})
                        token_count = comp_result.get('compressed_tokens', 0)
                    if token_count:
                        token_counts.append(token_count)
            
            if token_counts:
                row['Avg Tokens'] = int(sum(token_counts) / len(token_counts))
            
            # Add compression statistics if available
            if response_type != 'original':
                rate_values = stats[response_type]['compression'].get('rate_percent', [])
                ratio_values = stats[response_type]['compression'].get('ratio_x', [])
                if rate_values:
                    row['Compression Rate (%)'] = f"{(sum(rate_values)/len(rate_values)):.1f}"
                else:
                    row['Compression Rate (%)'] = 'N/A'
                if ratio_values:
                    row['Compression Ratio (x)'] = f"{(sum(ratio_values)/len(ratio_values)):.2f}"
                else:
                    row['Compression Ratio (x)'] = 'N/A'
            else:
                row['Compression Rate (%)'] = 'N/A'
                row['Compression Ratio (x)'] = 'N/A'
            
            table_data.append(row)
        
        # Create and display DataFrame
        df = pd.DataFrame(table_data)
        
        print("\n" + "=" * 100)
        print(f"EVALUATION RESULTS SUMMARY ({len(results)} examples)")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)
        
        # Save CSV if requested
        if csv_path is not None:
            try:
                csv_path = Path(csv_path)
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path, index=False)
                logger.info(f"Evaluation summary table saved to {csv_path}")
            except Exception as e:
                logger.error(f"Failed to save evaluation table CSV: {e}")
        
        return df

    def export_detailed_csv(self, results: List[Dict], csv_path: Path,
                            metrics: Optional[List[str]] = None):
        """Export a detailed per-example CSV including all response types.
        Columns: query_id, query, response_type, tokens, compression_ratio, compression_rate,
        and one column per metric (e.g., llm_judge, bleu, rouge).
        """
        if not results:
            logger.warning("No results to export")
            return
        
        # Determine metrics to include
        if metrics is None:
            metrics = set()
            for example in results:
                response_types = self.get_response_types(example)
                for response_type in response_types:
                    if 'evaluations' in example[response_type]:
                        metrics.update(example[response_type]['evaluations'].keys())
            metrics = sorted(list(metrics))
        
        # Helpers to parse compression fields
        def _parse_rate_to_percent(value: Any) -> Optional[float]:
            try:
                if value is None:
                    return None
                if isinstance(value, (int, float)):
                    return float(value) * 100.0 if float(value) <= 1.0 else float(value)
                s = str(value).strip()
                if s.endswith('%'):
                    return float(s[:-1])
                v = float(s)
                return v * 100.0 if v <= 1.0 else v
            except Exception:
                return None
        
        def _parse_ratio_to_x(value: Any) -> Optional[float]:
            try:
                if value is None:
                    return None
                if isinstance(value, (int, float)):
                    return float(value)
                s = str(value).strip().lower()
                if s.endswith('x'):
                    s = s[:-1]
                return float(s)
            except Exception:
                return None
        
        # Build rows
        rows: List[Dict[str, Any]] = []
        for example in results:
            query_id = example.get('query_id', '')
            query = example.get('query', '')
            response_types = self.get_response_types(example)
            # Include original as well
            if 'original' in example:
                response_types = ['original'] + response_types
            else:
                response_types = response_types
            
            for response_type in response_types:
                row: Dict[str, Any] = {
                    'query_id': query_id,
                    'query': query,
                    'response_type': response_type
                }
                
                if response_type == 'original':
                    tokens = example.get('original', {}).get('token_count', None)
                    row['tokens'] = tokens if tokens is not None else ''
                    row['compression_ratio'] = ''
                    row['compression_rate'] = ''
                else:
                    comp_result = example.get(response_type, {}).get('compression_result', {})
                    tokens = comp_result.get('compressed_tokens', None)
                    row['tokens'] = tokens if tokens is not None else ''
                    row['compression_ratio'] = _parse_ratio_to_x(comp_result.get('compression_ratio'))
                    rate_percent = _parse_rate_to_percent(comp_result.get('compression_rate'))
                    row['compression_rate'] = rate_percent
                
                # Metrics
                evals = example.get(response_type, {}).get('evaluations', {})
                for metric in metrics:
                    row[metric] = evals.get(metric, '')
                
                rows.append(row)
        
        # Write CSV
        try:
            csv_path = Path(csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            logger.info(f"Detailed evaluation rows saved to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save detailed CSV: {e}")