"""
Response Generator - Phase 2  
Handles response generation from compressed/original contexts
"""

import json
import logging
from typing import List, Dict, Any, Optional

from utils import BaseComponent, ScaleDownAPI


logger = logging.getLogger(__name__)


class ResponseGenerator(BaseComponent):
    """Handles response generation phase"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize API client
        api_config = self.config['api_config']
        self.api_client = ScaleDownAPI(
            api_key=api_config['api_key'],
            base_url=api_config['base_url'],
            model=api_config['model']
        )
    
    def get_response_types(self, example: Dict) -> List[str]:
        """Auto-detect response types from example"""
        response_types = []
        metadata_keys = {'query_id', 'query', 'ground_truth', 'contexts', 'is_selected'}
        
        for key in example.keys():
            if key not in metadata_keys and isinstance(example[key], dict):
                if 'context' in example[key]:
                    response_types.append(key)
        
        return response_types
    
    def has_responses(self, example: Dict) -> Dict[str, bool]:
        """Check which response types already have responses"""
        response_status = {}
        response_types = self.get_response_types(example)
        
        for response_type in response_types:
            if 'response' in example[response_type] and example[response_type]['response']:
                response_status[response_type] = True
            else:
                response_status[response_type] = False
        
        return response_status
    
    def generate_response_for_example(self, example: Dict, 
                                    response_types: Optional[List[str]] = None) -> Dict:
        """Generate responses for a single example"""
        # Auto-detect response types if not specified
        if response_types is None:
            response_types = self.get_response_types(example)
        
        # Check existing responses
        response_status = self.has_responses(example)
        
        query = example.get('query', '')
        
        # Generate responses for each specified type
        for response_type in response_types:
            if response_type in example and isinstance(example[response_type], dict):
                # Skip if response already exists
                if response_status.get(response_type, False):
                    logger.info(f"  Response for {response_type} already exists, skipping")
                    continue
                
                if 'context' in example[response_type]:
                    context = example[response_type]['context']
                    logger.info(f"  Generating response for {response_type}...")
                    
                    try:
                        response = self.api_client.get_response(context, query)
                        example[response_type]['response'] = response
                    except Exception as e:
                        logger.error(f"Error generating response for {response_type}: {e}")
                        example[response_type]['response'] = ""
                else:
                    logger.warning(f"No context found for {response_type}")
        
        return example
    
    def generate_from_file(self, input_file: str, 
                          response_types: Optional[List[str]] = None) -> List[Dict]:
        """Generate responses from input file"""
        logger.info(f"Loading data from {input_file}")
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        results = []
        for i, example in enumerate(data):
            logger.info(f"Generating responses for example {i+1}/{len(data)}: {example.get('query', '')[:50]}...")
            
            try:
                result = self.generate_response_for_example(example, response_types)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing example {i+1}: {e}")
                results.append(example)  # Keep original even if generation failed
                continue
        
        logger.info(f"Response generation completed. Processed {len(results)} examples")
        return results