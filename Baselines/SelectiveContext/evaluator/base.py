from abc import ABC, abstractmethod
from typing import Dict

class BaseEvaluator(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def evaluate(self, ground_truth: str, generated_response: str) -> Dict[str, float]:
        pass