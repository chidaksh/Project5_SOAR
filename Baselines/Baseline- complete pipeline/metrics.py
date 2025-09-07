"""
Evaluation Metrics - Supporting different evaluation methods
"""

import re
import logging
import subprocess
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge import Rouge
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK or rouge-score not available. BLEU and ROUGE metrics will not work.")

from utils import ScaleDownAPI


logger = logging.getLogger(__name__)


class LLMJudgeEvaluator:
    """LLM-as-a-Judge evaluator"""
    
    def __init__(self, api_client: ScaleDownAPI):
        self.api_client = api_client
    
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
            judge_response = self.api_client.get_response("", prompt, model=judge_model)
            
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


class BLEUEvaluator:
    """BLEU score evaluator"""
    
    def __init__(self):
        if not NLTK_AVAILABLE:
            raise ImportError("NLTK is required for BLEU evaluation. Install with: pip install nltk")
        self.smoothing = SmoothingFunction().method1
    
    def evaluate(self, ground_truth: str, generated_response: str) -> float:
        """Calculate BLEU score"""
        try:
            # Tokenize
            reference = ground_truth.lower().split()
            candidate = generated_response.lower().split()
            
            # Calculate BLEU score
            score = sentence_bleu([reference], candidate, smoothing_function=self.smoothing)
            return score
            
        except Exception as e:
            logger.error(f"BLEU evaluation error: {e}")
            return 0.0


class ROUGEEvaluator:
    """ROUGE score evaluator"""
    
    def __init__(self):
        if not NLTK_AVAILABLE:
            raise ImportError("rouge-score is required for ROUGE evaluation. Install with: pip install rouge-score")
        self.rouge = Rouge()
    
    def evaluate(self, ground_truth: str, generated_response: str) -> float:
        """Calculate ROUGE-L score"""
        try:
            if not ground_truth.strip() or not generated_response.strip():
                return 0.0
            
            scores = self.rouge.get_scores(generated_response, ground_truth)
            # Return ROUGE-L F1 score
            return scores[0]['rouge-l']['f']
            
        except Exception as e:
            logger.error(f"ROUGE evaluation error: {e}")
            return 0.0


class MSMARCOEvaluator:
    """MS-MARCO official evaluator wrapper"""
    
    def __init__(self, evaluator_path: Optional[str] = None):
        self.evaluator_path = evaluator_path
        # Try to find the official evaluator script
        if not evaluator_path:
            self.evaluator_path = self._find_evaluator_script()
    
    def _find_evaluator_script(self) -> Optional[str]:
        """Try to find MS-MARCO evaluator script"""
        possible_paths = [
            "eval.py",
            "msmarco_eval.py", 
            "MSMARCO-Question-Answering/Evaluation/eval.py"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        logger.warning("MS-MARCO evaluator script not found. Using fallback BLEU/ROUGE.")
        return None
    
    def evaluate(self, ground_truth: str, generated_response: str) -> Dict[str, float]:
        """Evaluate using MS-MARCO metrics"""
        if not self.evaluator_path:
            # Fallback to BLEU and ROUGE
            logger.info("Using BLEU/ROUGE fallback for MS-MARCO evaluation")
            return self._fallback_evaluate(ground_truth, generated_response)
        
        try:
            # Create temporary files for evaluation
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as ref_file:
                ref_file.write(ground_truth + '\n')
                ref_path = ref_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as cand_file:
                cand_file.write(generated_response + '\n')
                cand_path = cand_file.name
            
            # Run evaluator
            result = subprocess.run([
                'python', self.evaluator_path, ref_path, cand_path
            ], capture_output=True, text=True)
            
            # Parse results
            scores = self._parse_msmarco_output(result.stdout)
            
            # Cleanup
            Path(ref_path).unlink()
            Path(cand_path).unlink()
            
            return scores
            
        except Exception as e:
            logger.error(f"MS-MARCO evaluation error: {e}")
            return self._fallback_evaluate(ground_truth, generated_response)
    
    def _parse_msmarco_output(self, output: str) -> Dict[str, float]:
        """Parse MS-MARCO evaluator output"""
        scores = {}
        lines = output.strip().split('\n')
        
        for line in lines:
            if 'BLEU' in line:
                match = re.search(r'BLEU:\s*([\d.]+)', line)
                if match:
                    scores['bleu'] = float(match.group(1))
            elif 'ROUGE' in line:
                match = re.search(r'ROUGE-L:\s*([\d.]+)', line)
                if match:
                    scores['rouge_l'] = float(match.group(1))
        
        return scores
    
    def _fallback_evaluate(self, ground_truth: str, generated_response: str) -> Dict[str, float]:
        """Fallback evaluation using BLEU and ROUGE"""
        scores = {}
        
        try:
            # BLEU
            bleu_eval = BLEUEvaluator()
            scores['bleu'] = bleu_eval.evaluate(ground_truth, generated_response)
        except:
            scores['bleu'] = 0.0
        
        try:
            # ROUGE
            rouge_eval = ROUGEEvaluator()
            scores['rouge_l'] = rouge_eval.evaluate(ground_truth, generated_response)
        except:
            scores['rouge_l'] = 0.0
        
        return scores