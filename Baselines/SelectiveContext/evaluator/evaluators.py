"""
Evaluation Metrics - Supporting different evaluation methods
"""

import logging
from typing import Optional, Dict, Any
from generator.base import BaseGenerator
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge


from .base import BaseEvaluator

logger = logging.getLogger(__name__)

class LLMJudgeEvaluator:
    """LLM-as-a-Judge evaluator"""
    
    def __init__(self, generator: BaseGenerator):
        self.generator = generator
    
    def evaluate(self, ground_truth: str, generated_response: str, judge_model: Optional[str] = None) -> int:
        """Evaluate using LLM judge"""
        prompt = f"""Judge if the generated answer matches the ground truth or contains it correctly.

Ground Truth: {ground_truth}
Generated Answer: {generated_response}

Follow the below instructions strictly.
Instructions:
- If the generated answer captures the same meaning or key information as the ground truth, respond with "match".
- If the generated answer is incorrect, does not contain the ground truth, respond with "no match"
- Respond only with 'match' or 'no match' - no additional text."""
        
        try:
            judge_response = self.generator.generate_response("", prompt).get('generated_response')
            
            if judge_response:
                judge_response = judge_response.strip().lower()
                if "match" in judge_response and "no match" not in judge_response:
                    return 1
                elif "no match" in judge_response:
                    return 0
            
            return 0  # Default to 0 if unclear
            
        except Exception as e:
            logger.error(f"LLM Judge evaluation error: {e}")
            return 0


class BLEUEvaluator(BaseEvaluator):
    """BLEU score evaluator"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smoothing_function = SmoothingFunction().method1
    
    def evaluate(self, ground_truth: str, generated_response: str) -> Dict[str, float]:
        """Calculate BLEU score"""
        
        try:
            # Tokenize the texts
            reference_tokens = ground_truth.lower().split()
            candidate_tokens = generated_response.lower().split()
            
            # Calculate BLEU score
            if len(candidate_tokens) == 0:
                return {"bleu": 0.0}
            
            # Use smoothing to handle edge cases
            bleu_score = sentence_bleu(
                [reference_tokens], 
                candidate_tokens,
                smoothing_function=self.smoothing_function
            )
            
            return {"bleu": bleu_score}
            
        except Exception as e:
            logger.error(f"BLEU evaluation error: {e}")
            return {"bleu": 0.0}


class ROUGEEvaluator(BaseEvaluator):
    """ROUGE score evaluator"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.rouge = Rouge()
    
    def evaluate(self, ground_truth: str, generated_response: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        
        try:
            # Handle empty responses
            if not generated_response.strip():
                return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
            
            # Calculate ROUGE scores
            scores = self.rouge.get_scores(generated_response, ground_truth)[0]
            
            return {
                "rouge_1": scores['rouge-1']['f'],
                "rouge_2": scores['rouge-2']['f'],
                "rouge_l": scores['rouge-l']['f']
            }
            
        except Exception as e:
            logger.error(f"ROUGE evaluation error: {e}")
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}


# class MSMARCOEvaluator(BaseEvaluator):
#     """MS-MARCO official evaluator wrapper"""
    
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__(config)
#         self.evaluator_path = config['evaluation'].get("msmarcco_repo_path")
#         if not self.evaluator_path:
#             raise ValueError("msmarcco_repo_path path is required")
    
#     def evaluate(self, ground_truth: str, generated_response: str) -> Dict[str, float]:
#         """Evaluate using MS-MARCO metrics"""
#         if not self.evaluator_path:
#             # Fallback to BLEU and ROUGE
#             logger.info("Using BLEU/ROUGE fallback for MS-MARCO evaluation")
#             return self._fallback_evaluate(ground_truth, generated_response)
        
#         try:
#             # Create temporary files for evaluation
#             import tempfile
#             with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as ref_file:
#                 ref_file.write(ground_truth + '\n')
#                 ref_path = ref_file.name
            
#             with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as cand_file:
#                 cand_file.write(generated_response + '\n')
#                 cand_path = cand_file.name
            
#             # Run evaluator
#             result = subprocess.run([
#                 'python', self.evaluator_path, ref_path, cand_path
#             ], capture_output=True, text=True)

#             print(f"python {self.evaluator_path} {ref_path} {cand_path}")
            
#             # Parse results
#             scores = self._parse_msmarco_output(result.stdout)
#             print("STDOUT: ", result.stderr)
            
#             # Cleanup
#             Path(ref_path).unlink()
#             Path(cand_path).unlink()
            
#             return scores
            
#         except Exception as e:
#             logger.error(f"MS-MARCO evaluation error: {e}")
#             return self._fallback_evaluate(ground_truth, generated_response)
    
#     def _parse_msmarco_output(self, output: str) -> Dict[str, float]:
#         """Parse MS-MARCO evaluator output"""
#         scores = {}
#         lines = output.strip().split('\n')
        
#         for line in lines:
#             if 'BLEU' in line:
#                 match = re.search(r'BLEU:\s*([\d.]+)', line)
#                 if match:
#                     scores['bleu'] = float(match.group(1))
#             elif 'ROUGE' in line:
#                 match = re.search(r'ROUGE-L:\s*([\d.]+)', line)
#                 if match:
#                     scores['rouge_l'] = float(match.group(1))
        
#         return scores
    
#     def _fallback_evaluate(self, ground_truth: str, generated_response: str) -> Dict[str, float]:
#         """Fallback evaluation using BLEU and ROUGE"""
#         scores = {}
        
#         try:
#             # BLEU
#             bleu_eval = BLEUEvaluator(self.config)
#             bleu_result = bleu_eval.evaluate(ground_truth, generated_response)
#             scores.update(bleu_result)
#         except:
#             scores['bleu'] = 0.0
        
#         try:
#             # ROUGE
#             rouge_eval = ROUGEEvaluator(self.config)
#             rouge_result = rouge_eval.evaluate(ground_truth, generated_response)
#             scores.update(rouge_result)
#         except:
#             scores['rouge_l'] = 0.0
        
#         return scores