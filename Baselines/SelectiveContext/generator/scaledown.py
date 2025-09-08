import json
import logging
import requests
from datetime import datetime
from generator.base import BaseGenerator

class ScaleDownAPI:
    """API client for ScaleDown service"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def get_response(self, context: str, prompt: str) -> str:
        """Get response from API"""
        payload = {
            "context": context,
            "model": self.model,
            "scaledown": {"rate": 0},
            "prompt": prompt
        }

        headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json'
        }

        try:
            response = requests.post(self.base_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json().get('full_response', '')
        except Exception as e:
            logging.error(f"API Error: {e}")
            return ""

class ScaleDownGenerator(BaseGenerator):
    """ScaleDown API implementation of BaseGenerator"""
    
    def __init__(self, config):
        super().__init__(config, config['generation']['api']['model'])
        
        api_cfg = config['generation']['api']
        self.api = ScaleDownAPI(
            api_key = api_cfg['api_key'],
            base_url = api_cfg['base_url'],
            model = api_cfg['model'],
        )
    
    def generate_response(self, query: str, context: str) -> dict:
        """Generate response using ScaleDown API"""
        try:
            # Format context from compressed passages
            response_text = self.api.get_response(context, query)

            return {
                'generated_response': response_text,
                'generation_metadata': {
                    'generator': self.get_generator_name(),
                    'model': self.model,
                    'timestamp': datetime.now().isoformat(),
                    'generation_error': ''
                }
            }
            
        except Exception as e:
            return {
                'generated_response': '',
                'generation_metadata': {
                    'generator': self.get_generator_name(),
                    'model': self.model,
                    'timestamp': datetime.now().isoformat(),
                    'generation_error': str(e)
                }
            }
    
    def get_generator_name(self):
        return "ScaleDownAPI"