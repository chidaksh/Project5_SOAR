from abc import ABC, abstractmethod
class BaseGenerator(ABC):
    """Abstract base class for response generators"""
    
    def __init__(self, config, model: str):
        self.config = config
        self.model = model # logging purpose
    
    @abstractmethod
    def generate_response(self, query: str, context: str) -> dict:
        """Generate response for given query and compressed passages"""
        pass
    
    @abstractmethod
    def get_generator_name(self):
        """Return generator name for metadata"""
        pass