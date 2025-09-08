import os
import json
from datetime import datetime
from typing import Dict, Any, List
from tqdm import tqdm
from evaluator import LLMJudgeEvaluator, BLEUEvaluator, ROUGEEvaluator
from generator.scaledown import ScaleDownGenerator
from utils import get_dataset_path, setup_experiment, load_compressed_dataset


def _load_generated_dataset(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load dataset with generated responses"""
    dataset = load_compressed_dataset(config)
    
    # Filter samples that have generated responses
    generated_samples = [s for s in dataset if s.get('generated_response')]
    print(f"Found {len(generated_samples)} samples with generated responses")
    return generated_samples


def _create_evaluators(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create evaluation metrics based on config"""
    evaluators = {}
    
    eval_config = config.get('evaluation', {})
    metrics = eval_config.get('metrics', ['bleu', 'rouge', 'msmarco'])
    
    if 'bleu' in metrics:
        evaluators['bleu'] = BLEUEvaluator(config)
    if 'rouge' in metrics:
        evaluators['rouge'] = ROUGEEvaluator(config)
    if 'msmarco' in metrics:
        evaluators['msmarco'] = MSMARCOEvaluator(config)
    if 'llm_judge' in metrics:
        generator = ScaleDownGenerator(config)
        evaluators['llm_judge'] = LLMJudgeEvaluator(generator)
    
    return evaluators


def evaluate_responses(config: Dict[str, Any]) -> None:
    """Main evaluation function"""
    print("Starting evaluation...")
    
    # Load dataset with generated responses
    samples = _load_generated_dataset(config)
    if not samples:
        print("No samples with generated responses found.")
        return
    
    # Create evaluators
    evaluators = _create_evaluators(config)
    print(f"Using evaluators: {list(evaluators.keys())}")
    
    # Evaluate each sample
    results = []
    for sample in tqdm(samples, desc="Evaluating"):
        try:
            ground_truth = " ".join(sample['answers']) if sample['answers'] else ""
            generated_response = sample.get('generated_response', "")
            
            sample_results = {
                'query_id': sample['query_id'],
                'query': sample['query'],
                'ground_truth': ground_truth,
                'generated_response': generated_response
            }
            
            # Run each evaluator
            for name, evaluator in evaluators.items():
                if name == 'llm_judge':
                    score = evaluator.evaluate(ground_truth, generated_response)
                    sample_results['llm_judge_score'] = score
                else:
                    scores = evaluator.evaluate(ground_truth, generated_response)
                    sample_results.update(scores)
            
            results.append(sample_results)
            
        except Exception as e:
            print(f"Error evaluating sample {sample.get('query_id', 'unknown')}: {e}")
    
    # Calculate aggregate metrics
    aggregate_metrics = _calculate_aggregate_metrics(results)
    
    # Save results
    _save_evaluation_results(config, results, aggregate_metrics)
    
    print(f"Evaluation completed. Processed {len(results)} samples.")
    print("Aggregate metrics:", {k: f"{v:.3f}" for k, v in aggregate_metrics.items()})


def _calculate_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate aggregate metrics from individual results"""
    if not results:
        return {}
    
    metrics = {}
    metric_keys = [k for k in results[0].keys() if k not in ['query_id', 'query', 'ground_truth', 'generated_response']]
    
    for key in metric_keys:
        values = [r[key] for r in results if key in r and r[key] is not None]
        if values:
            metrics[f"avg_{key}"] = sum(values) / len(values)
            metrics[f"count_{key}"] = len(values)
    
    return metrics


def _save_evaluation_results(config: Dict[str, Any], results: List[Dict[str, Any]], aggregate_metrics: Dict[str, float]) -> None:
    """Save evaluation results to JSON file"""
    # Save detailed results
    results_path = get_dataset_path(config, "_evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'evaluation_results': results,
            'aggregate_metrics': aggregate_metrics,
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(results),
                'metrics_used': list(aggregate_metrics.keys())
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation results saved to: {results_path}")


def main() -> None:
    config = setup_experiment("evaluation.py")
    evaluate_responses(config)


if __name__ == "__main__":
    main()