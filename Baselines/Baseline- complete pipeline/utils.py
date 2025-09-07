"""
Utility classes and functions for the LLMLingua2 pipeline
"""

import json
import logging
import requests
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Try to import datasets library, set flag if not available
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # Force reconfiguration
    )
    
    # Set level for all existing loggers
    for logger_name in ['compressor', 'generator', 'evaluator', 'utils']:
        logging.getLogger(logger_name).setLevel(level)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def validate_input_file(file_path: str) -> bool:
    """Validate input file exists and is valid JSON"""
    try:
        path = Path(file_path)
        if not path.exists():
            return False
        
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except Exception:
        return False

def load_external_json_dataset(file_path: str) -> List[Dict]:
    """Load external JSON dataset"""
    logging.info(f"Loading external dataset from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def load_msmarco_dataset(config: Dict[str, Any], external_dataset_path: Optional[str] = None) -> List[Dict]:
    """
    Load and filter MS MARCO dataset or external dataset if provided
    
    Args:
        config: Configuration dictionary containing dataset_config
        external_dataset_path: Path to external JSON dataset (optional)
    
    Returns:
        List of filtered examples
    """
    if external_dataset_path is not None:
        # Load external dataset
        return load_external_json_dataset(external_dataset_path)
    
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library is required for MS-MARCO loading. Install with: pip install datasets")
    
    # Load MS MARCO dataset
    dataset_config = config.get('dataset_config', {})
    version = dataset_config.get('version', 'v2.1')
    query_type = dataset_config.get('query_type', 'NUMERIC')
    max_examples = dataset_config.get('max_examples', 100)
    start = dataset_config.get('start', 0)
    
    logging.info("Loading MS MARCO dataset...")
    dataset = load_dataset('microsoft/ms_marco', version)['validation']
    
    numeric_example_count = 0
    filtered_examples = []
    
    for example in dataset:
        # Stop if we have enough examples
        if len(filtered_examples) >= max_examples:
            break
        
        # Filter by query type
        if example['query_type'] != query_type:
            continue
        
        # Skip examples without answers
        if not example['answers'] or not example['answers'][0]:
            continue
        
        # Skip "no answer" responses
        if example['answers'][0].lower().strip() in ['no answer', 'no answer present', 'no answer present.']:
            continue
        
        numeric_example_count += 1
        
        # Skip examples before start index
        if numeric_example_count < start:
            continue
        
        filtered_examples.append(example)
    
    logging.info(f"Loaded {len(filtered_examples)} examples")
    return filtered_examples

def normalize_data_format(data: List[Dict]) -> List[Dict]:
    """
    Normalize different data formats to a consistent structure
    
    Handles both MS-MARCO format and custom formats
    """
    normalized = []
    
    for example in data:
        # Check if it's MS-MARCO format
        if 'passages' in example:
            # MS-MARCO format
            normalized_example = {
                'query_id': example['query_id'],
                'query': example['query'],
                'ground_truth': example['answers'][0] if example['answers'] else '',
                'contexts': example['passages']['passage_text'],
                'is_selected': example['passages']['is_selected']
            }
        else:
            # Already in normalized format or custom format
            normalized_example = {
                'query_id': example.get('query_id', ''),
                'query': example.get('query', ''),
                'ground_truth': example.get('ground_truth', ''),
                'contexts': example.get('contexts', []),
                'is_selected': example.get('is_selected', [])
            }
        
        normalized.append(normalized_example)
    
    return normalized

def load_and_prepare_dataset(config: Dict[str, Any], 
                           external_dataset_path: Optional[str] = None,
                           num_examples: Optional[int] = None) -> List[Dict]:
    """
    Complete dataset loading and preparation pipeline
    
    Args:
        config: Configuration dictionary
        external_dataset_path: Path to external dataset (optional)
        num_examples: Limit number of examples (optional)
    
    Returns:
        List of prepared examples in normalized format
    """
    # Load raw data
    data = load_msmarco_dataset(config, external_dataset_path)
    
    # Normalize format
    normalized_data = normalize_data_format(data)
    
    # Apply example limit if specified
    if num_examples:
        normalized_data = normalized_data[:num_examples]
        logging.info(f"Limited to {num_examples} examples")
    
    return normalized_data

class BaseComponent:
    """Base class for pipeline components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logging.info(f"Results saved to {output_file}")

    def load_dataset(self, external_dataset_path: Optional[str] = None, 
                    num_examples: Optional[int] = None) -> List[Dict]:
        """Load dataset using the utility function"""
        return load_and_prepare_dataset(
            self.config, 
            external_dataset_path=external_dataset_path,
            num_examples=num_examples
        )    


class ContextTracker:
    """Track context retention with separators"""
    
    def __init__(self, separator: str):
        self.separator = separator
    
    def prepare_contexts_with_separators(self, contexts: List[str]) -> Tuple[str, Dict[int, Tuple[int, int]]]:
        """Combine contexts with separators and track positions"""
        combined_text = ""
        context_positions = {}
        current_pos = 0
        
        for i, context in enumerate(contexts):
            if i > 0:
                combined_text += f" {self.separator} "
                current_pos += len(f" {self.separator} ")
            
            start_pos = current_pos
            combined_text += context
            current_pos += len(context)
            end_pos = current_pos
            
            context_positions[i] = (start_pos, end_pos)
            
        return combined_text, context_positions
    
    def analyze_context_retention(self, original_text: str, compressed_text: str,
                                context_positions: Dict[int, Tuple[int, int]]) -> Dict[int, Dict[str, Any]]:
        """Analyze which tokens from each context are retained"""
        cleaned_compressed = compressed_text.replace(self.separator, " ").strip()
        compressed_words = cleaned_compressed.split()
        
        context_stats = {}
        
        for context_id, (start, end) in context_positions.items():
            original_context_text = original_text[start:end]
            original_words = original_context_text.split()
            
            retained_count = 0
            for word in original_words:
                if word.lower() in [w.lower() for w in compressed_words]:
                    retained_count += 1
            
            context_stats[context_id] = {
                'original_length': len(original_words),
                'retained_count': retained_count,
                'retention_ratio': retained_count / max(len(original_words), 1)
            }
        
        return context_stats


class ScaleDownAPI:
    """API client for ScaleDown service"""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
    
    def get_response(self, context: str, prompt: str, model: Optional[str] = None) -> str:
        """Get response from API"""
        payload = {
            "context": context,
            "model": model if model else self.model,
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
            print(response.text)
            print(response.json().get('full_response'))
            return response.json().get('full_response')
        except Exception as e:
            logging.error(f"API Error: {e}")
            return ""