"""
Compression Pipeline - Phase 1
Handles prompt compression using LLMLingua2 methods
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from llmlingua import PromptCompressor

from utils import BaseComponent, ContextTracker, load_and_prepare_dataset


logger = logging.getLogger(__name__)


class CompressionPipeline(BaseComponent):
    """Handles prompt compression phase"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize context tracker
        self.context_tracker = ContextTracker(self.config['context_separator'])
        
        # Initialize LLMLingua2 compressor
        logger.info("Initializing LLMLingua2 compressor...")
        self.compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="auto"
        )
        
        # Get compression methods from config
        self.compression_methods = self.config.get('compression_methods', {})
        
        # Configure chunking to support long contexts
        # Allows the compressor to split inputs on these token boundaries
        default_chunk_tokens = [
            self.config['context_separator'], '\n', '.', '?', '!', ';', ':'
        ]
        self.chunk_end_tokens = self.config.get('chunk_end_tokens', default_chunk_tokens)
        
    def compress_with_method(self, contexts: List[str], query: str, method_config: Dict) -> Dict:
        """Apply specific compression method"""
        combined_context, context_positions = self.context_tracker.prepare_contexts_with_separators(contexts)
        
        # Let LLMLingua2 handle long context via chunking; avoid manual truncation
        
        force_tokens = [self.config['context_separator'], '\n', '.', '?']
        
        try:
            if 'rate' in method_config:
                result = self.compressor.compress_prompt(
                    context=[combined_context],
                    question=query,
                    rate=method_config['rate'],
                    force_tokens=force_tokens,
                    use_token_level_filter=True,
                    chunk_end_tokens=self.chunk_end_tokens
                )
            elif 'target_token' in method_config:
                result = self.compressor.compress_prompt(
                    context=[combined_context],
                    question=query,
                    target_token=method_config['target_token'],
                    force_tokens=force_tokens,
                    use_token_level_filter=True,
                    chunk_end_tokens=self.chunk_end_tokens
                )
            elif 'target_context' in method_config:
                result = self.compressor.compress_prompt(
                    context=contexts,
                    question=query,
                    target_context=method_config['target_context'],
                    force_tokens=force_tokens,
                    use_context_level_filter=True,
                    use_token_level_filter=True,
                    chunk_end_tokens=self.chunk_end_tokens
                )
            else:
                raise ValueError(f"Invalid compression method config: {method_config}")
            
            # Analyze context retention
            context_analysis = self.context_tracker.analyze_context_retention(
                combined_context, result['compressed_prompt'], context_positions
            )
            
            return {
                'compressed_prompt': result['compressed_prompt'],
                'compression_rate': result['rate'],
                'compression_ratio': result['ratio'],
                'original_tokens': result['origin_tokens'],
                'compressed_tokens': result['compressed_tokens'],
                'context_analysis': context_analysis
            }
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return {
                'compressed_prompt': combined_context,
                'compression_rate': "100%",
                'compression_ratio': "1.0x",
                'original_tokens': len(combined_context.split()),
                'compressed_tokens': len(combined_context.split()),
                'context_analysis': {},
                'error': str(e)
            }
    
    def compress_example(self, example: Dict, compression_methods: Optional[List[str]] = None) -> Dict:
        """Compress a single example"""
        '''# Handle different input formats
        if 'passages' in example:  # MS MARCO format
            contexts = example['passages']['passage_text']
            query = example['query']
            ground_truth = example['answers'][0]
            query_id = example['query_id']
            is_selected = example['passages']['is_selected']
        else:  # Custom format
            contexts = example.get('contexts', [])
            query = example.get('query', '')
            ground_truth = example.get('ground_truth', '')
            query_id = example.get('query_id', '')
            is_selected = example.get('is_selected', [])'''
        # Data is already normalized by load_and_prepare_dataset
        contexts = example.get('contexts', [])
        query = example.get('query', '')
        ground_truth = example.get('ground_truth', '')
        query_id = example.get('query_id', '')
        is_selected = example.get('is_selected', [])

        result = {
            'query_id': query_id,
            'query': query,
            'ground_truth': ground_truth,
            'contexts': contexts,
            'is_selected': is_selected
        }
        
        # Add original context
        original_context = "\n\n".join(contexts)
        result['original'] = {
            'context': original_context,
            'token_count': len(original_context.split())
        }
        
        # Determine which methods to use
        methods_to_use = compression_methods if compression_methods else list(self.compression_methods.keys())
        
        # Apply compression methods
        for method_name in methods_to_use:
            if method_name not in self.compression_methods:
                logger.warning(f"Method {method_name} not found in config")
                continue
                
            logger.info(f"  Compressing with {method_name}...")
            method_config = self.compression_methods[method_name]
            compression_result = self.compress_with_method(contexts, query, method_config)
            
            result[method_name] = {
                'compression_result': compression_result,
                'context': compression_result['compressed_prompt']
            }
        
        return result
    
    def compress_from_file(self, input_file: str, compression_methods: Optional[List[str]] = None,
                          num_examples: Optional[int] = None) -> List[Dict]:
        """Compress examples from input file"""
        logger.info(f"Loading data from {input_file}")
        
        # Use utility function for consistent data loading
        data = load_and_prepare_dataset(self.config, external_dataset_path=input_file, num_examples=num_examples)
        
        '''with open(input_file, 'r') as f:
            data = json.load(f)'''
        
        '''if num_examples:
            data = data[:num_examples]
            logger.info(f"Processing {num_examples} examples")'''
        
        results = []
        for i, example in enumerate(data):
            logger.info(f"Processing example {i+1}/{len(data)}: {example.get('query', '')[:50]}...")
            
            try:
                result = self.compress_example(example, compression_methods)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing example {i+1}: {e}")
                continue
        
        logger.info(f"Compression completed. Processed {len(results)} examples")
        return results
    def compress_from_msmarco(self, compression_methods: Optional[List[str]] = None,
                             num_examples: Optional[int] = None) -> List[Dict]:
        """Compress examples directly from MS-MARCO dataset"""
        logger.info("Loading MS-MARCO dataset...")
        
        # Use utility function to load MS-MARCO
        data = load_and_prepare_dataset(self.config, external_dataset_path=None, num_examples=num_examples)
        
        results = []
        for i, example in enumerate(data):
            logger.info(f"Processing example {i+1}/{len(data)}: {example.get('query', '')[:50]}...")
            
            try:
                result = self.compress_example(example, compression_methods)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing example {i+1}: {e}")
                continue
        
        logger.info(f"Compression completed. Processed {len(results)} examples")
        return results      