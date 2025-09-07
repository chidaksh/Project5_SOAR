#!/usr/bin/env python3
"""
LLMLingua2 Pipeline - Comprehensive Usage Examples
Demonstrates various ways to use the modular system with both custom data and MS-MARCO dataset
"""

import json
from pathlib import Path
from typing import List, Dict, Any

def create_sample_data() -> List[Dict[str, Any]]:
    """Create sample input data for testing"""
    sample_data = [
        {
            "query_id": 1,
            "query": "What is the capital of France?",
            "ground_truth": "Paris is the capital of France.",
            "contexts": [
                "France is a country in Western Europe. Its capital and largest city is Paris.",
                "Paris, the capital of France, is known for the Eiffel Tower and the Louvre Museum.",
                "The Seine river flows through Paris, which is the political and cultural center of France."
            ],
            "is_selected": [1, 1, 0]
        },
        {
            "query_id": 2, 
            "query": "How many days are in February during a leap year?",
            "ground_truth": "February has 29 days during a leap year.",
            "contexts": [
                "A leap year occurs every 4 years to account for the extra time it takes Earth to orbit the sun.",
                "February typically has 28 days, but during leap years it has 29 days.",
                "Leap years are divisible by 4, except for years divisible by 100 unless also divisible by 400."
            ],
            "is_selected": [0, 1, 0]
        }
    ]
    
    with open('sample_data.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("✓ Sample data created: sample_data.json")
    return sample_data


def example_msmarco_usage():
    """Example: Using MS-MARCO dataset directly"""
    print("="*80)
    print("EXAMPLE 1: MS-MARCO Dataset Usage")
    print("="*80)
    
    print("The system can load MS-MARCO dataset directly without requiring input files:")
    print()
    
    # MS-MARCO examples
    msmarco_examples = [
        {
            'name': 'Full Pipeline with MS-MARCO',
            'command': [
                'python', 'pipeline.py',
                '--phases', 'all',
                '--use-msmarco',
                '--config', 'config.json',
                '--verbose'
            ],
            'description': 'Run complete pipeline using MS-MARCO dataset'
        },
        {
            'name': 'Compression Only with MS-MARCO',
            'command': [
                'python', 'pipeline.py',
                '--phases', 'compression',
                '--use-msmarco',
                '--output', 'msmarco_compressed.json',
                '--compression-methods', 'rate_based', 'token_based',
                '--num-examples', '50',
                '--verbose'
            ],
            'description': 'Compress MS-MARCO examples with specific methods'
        },
        {
            'name': 'MS-MARCO with Custom Configuration',
            'command': [
                'python', 'pipeline.py',
                '--phases', 'compression',
                '--use-msmarco',
                '--config', 'msmarco_config.json',
                '--num-examples', '100'
            ],
            'description': 'Use custom configuration for MS-MARCO processing'
        }
    ]
    
    for example in msmarco_examples:
        print(f"\n{example['name']}:")
        print(f"  Description: {example['description']}")
        print(f"  Command: {' '.join(example['command'])}")
    
    print(f"\nKey Benefits:")
    print(f"  • No need to download MS-MARCO manually")
    print(f"  • Automatic filtering by query type")
    print(f"  • Configurable dataset parameters")
    print(f"  • Built-in data normalization")


def example_custom_data_usage():
    """Example: Using custom input data"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Data Usage")
    print("="*80)
    
    # Create sample data
    create_sample_data()
    
    custom_examples = [
        {
            'name': 'Full Pipeline with Custom Data',
            'command': [
                'python', 'pipeline.py',
                '--phases', 'all',
                '--input', 'sample_data.json',
                '--config', 'config.json',
                '--verbose'
            ],
            'description': 'Run complete pipeline with custom input file'
        },
        {
            'name': 'Phase-by-Phase Execution',
            'command': [
                'python', 'pipeline.py',
                '--phases', 'compression',
                '--input', 'sample_data.json',
                '--output', 'compressed.json',
                '--compression-methods', 'rate_based', 'context_based'
            ],
            'description': 'Run compression phase only'
        },
        {
            'name': 'Generation from Compressed Data',
            'command': [
                'python', 'pipeline.py',
                '--phases', 'generation',
                '--input', 'compressed.json',
                '--output', 'responses.json'
            ],
            'description': 'Generate responses from compressed data'
        },
        {
            'name': 'Evaluation with Multiple Metrics',
            'command': [
                'python', 'pipeline.py',
                '--phases', 'evaluation',
                '--input', 'responses.json',
                '--output', 'final_results.json',
                '--metrics', 'llm_judge', 'bleu', 'rouge',
                '--export-msmarco'
            ],
            'description': 'Evaluate with multiple metrics and export MS-MARCO format'
        }
    ]
    
    for example in custom_examples:
        print(f"\n{example['name']}:")
        print(f"  Description: {example['description']}")
        print(f"  Command: {' '.join(example['command'])}")


def example_advanced_configurations():
    """Example: Advanced configuration options"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Advanced Configuration Options")
    print("="*80)
    
    # Create different configuration files
    configs = {
        "basic_config.json": {
            "context_separator": "<<<>>>",
            "api_config": {
                "api_key": "your_api_key_here",
                "base_url": "https://api.scaledown.xyz/compress/",
                "model": "gemini/gemini-2.0-flash"
            },
            "compression_methods": {
                "rate_based": {"rate": 0.5}
            },
            "evaluation": {
                "enabled": True,
                "metric": "llm_judge"
            },
            "dataset_config": {
                "version": "v2.1",
                "query_type": "NUMERIC",
                "max_examples": 50,
                "start": 0
            }
        },
        
        "advanced_config.json": {
            "context_separator": "<<<>>>",
            "api_config": {
                "api_key": "your_api_key_here",
                "base_url": "https://api.scaledown.xyz/compress/",
                "model": "gemini/gemini-2.0-flash"
            },
            "compression_methods": {
                "rate_based": {"rate": 0.3},
                "token_based": {"target_token": 150},
                "context_based": {"target_context": 2}
            },
            "evaluation": {
                "enabled": True,
                "metric": "llm_judge",
                "parameters": {
                    "judge_model": "gemini/gemini-1.5-pro",
                    "threshold": 0.8
                }
            },
            "dataset_config": {
                "version": "v2.1",
                "query_type": "NUMERIC",
                "max_examples": 100,
                "start": 0
            }
        },
        
        "msmarco_focused_config.json": {
            "context_separator": "<<<>>>",
            "api_config": {
                "api_key": "your_api_key_here",
                "base_url": "https://api.scaledown.xyz/compress/",
                "model": "gemini/gemini-2.0-flash"
            },
            "compression_methods": {
                "rate_based": {"rate": 0.2},
                "token_based": {"target_token": 100},
                "context_based": {"target_context": 1}
            },
            "evaluation": {
                "enabled": True,
                "metric": "llm_judge",
                "parameters": {
                    "judge_model": "gemini/gemini-1.5-pro",
                    "threshold": 0.7
                }
            },
            "dataset_config": {
                "version": "v2.1",
                "query_type": "NUMERIC",
                "max_examples": 200,
                "start": 0
            }
        }
    }
    
    for filename, config in configs.items():
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Created {filename}")
    
    print(f"\nConfiguration Examples:")
    print(f"  • basic_config.json: Simple setup with rate-based compression")
    print(f"  • advanced_config.json: Full feature set with all compression methods")
    print(f"  • msmarco_focused_config.json: Optimized for MS-MARCO dataset")
    
    print(f"\nUsage with different configs:")
    print(f"  python pipeline.py --phases all --use-msmarco --config basic_config.json")
    print(f"  python pipeline.py --phases compression --use-msmarco --config advanced_config.json")


def example_batch_processing():
    """Example: Processing large datasets in batches"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch Processing for Large Datasets")
    print("="*80)
    
    print("For large datasets, process in phases and batches to manage resources:")
    print()
    
    batch_workflow = [
        {
            'step': 'Step 1: Compression (GPU Machine)',
            'commands': [
                'python pipeline.py --phases compression --use-msmarco --num-examples 100 --output batch1_compressed.json --verbose',
                'python pipeline.py --phases compression --use-msmarco --num-examples 100 --output batch2_compressed.json --verbose',
                'python pipeline.py --phases compression --use-msmarco --num-examples 100 --output batch3_compressed.json --verbose'
            ],
            'description': 'Run compression on GPU machine with different batches'
        },
        {
            'step': 'Step 2: Generation (CPU Machine)',
            'commands': [
                'python pipeline.py --phases generation --input batch1_compressed.json --output batch1_responses.json',
                'python pipeline.py --phases generation --input batch2_compressed.json --output batch2_responses.json',
                'python pipeline.py --phases generation --input batch3_compressed.json --output batch3_responses.json'
            ],
            'description': 'Run generation on CPU machine (API calls)'
        },
        {
            'step': 'Step 3: Evaluation (Any Machine)',
            'commands': [
                'python pipeline.py --phases evaluation --input batch1_responses.json --output batch1_final.json --export-msmarco',
                'python pipeline.py --phases evaluation --input batch2_responses.json --output batch2_final.json --export-msmarco',
                'python pipeline.py --phases evaluation --input batch3_responses.json --output batch3_final.json --export-msmarco'
            ],
            'description': 'Run evaluation on any machine'
        }
    ]
    
    for workflow in batch_workflow:
        print(f"\n{workflow['step']}:")
        print(f"  {workflow['description']}")
        for cmd in workflow['commands']:
            print(f"  {cmd}")
    
    print(f"\nBatch Processing Benefits:")
    print(f"  • Manage memory usage with smaller batches")
    print(f"  • Distribute work across different machines")
    print(f"  • Resume processing if interrupted")
    print(f"  • Parallel processing of different batches")


def example_evaluation_scenarios():
    """Example: Different evaluation scenarios"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Evaluation Scenarios")
    print("="*80)
    
    evaluation_scenarios = [
        {
            'name': 'Quick Evaluation',
            'command': [
                'python', 'pipeline.py',
                '--phases', 'evaluation',
                '--input', 'responses.json',
                '--metrics', 'llm_judge'
            ],
            'description': 'Fast evaluation with LLM judge only'
        },
        {
            'name': 'Comprehensive Evaluation',
            'command': [
                'python', 'pipeline.py',
                '--phases', 'evaluation',
                '--input', 'responses.json',
                '--metrics', 'llm_judge', 'bleu', 'rouge',
                '--export-msmarco'
            ],
            'description': 'Full evaluation with all metrics and MS-MARCO export'
        },
        {
            'name': 'MS-MARCO Standard Evaluation',
            'command': [
                'python', 'pipeline.py',
                '--phases', 'evaluation',
                '--input', 'responses.json',
                '--metrics', 'llm_judge',
                '--export-msmarco'
            ],
            'description': 'Standard MS-MARCO evaluation format'
        },
        {
            'name': 'Research Evaluation',
            'command': [
                'python', 'pipeline.py',
                '--phases', 'evaluation',
                '--input', 'responses.json',
                '--metrics', 'bleu', 'rouge',
                '--verbose'
            ],
            'description': 'Research-focused evaluation with detailed logging'
        }
    ]
    
    for scenario in evaluation_scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Description: {scenario['description']}")
        print(f"  Command: {' '.join(scenario['command'])}")


def example_logging_and_debugging():
    """Example: Logging and debugging options"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Logging and Debugging")
    print("="*80)
    
    logging_examples = [
        {
            'name': 'Basic Logging',
            'command': [
                'python', 'pipeline.py',
                '--phases', 'compression',
                '--use-msmarco',
                '--num-examples', '10'
            ],
            'description': 'Standard logging output'
        },
        {
            'name': 'Verbose Debugging',
            'command': [
                'python', 'pipeline.py',
                '--phases', 'all',
                '--use-msmarco',
                '--num-examples', '5',
                '--verbose'
            ],
            'description': 'Detailed debugging information'
        },
        {
            'name': 'Debug Single Example',
            'command': [
                'python', 'pipeline.py',
                '--phases', 'compression',
                '--use-msmarco',
                '--num-examples', '1',
                '--verbose'
            ],
            'description': 'Debug with single example for troubleshooting'
        }
    ]
    
    for example in logging_examples:
        print(f"\n{example['name']}:")
        print(f"  Description: {example['description']}")
        print(f"  Command: {' '.join(example['command'])}")
    
    print(f"\nLogging Output Levels:")
    print(f"  • INFO: General progress information")
    print(f"  • DEBUG: Detailed debugging (use --verbose)")
    print(f"  • WARNING: Non-critical issues")
    print(f"  • ERROR: Critical errors")
    
    print(f"\nDebugging Tips:")
    print(f"  • Use --num-examples 1 for single example debugging")
    print(f"  • Check GPU memory with --verbose for compression issues")
    print(f"  • Monitor API calls with --verbose for generation/evaluation")


def example_performance_optimization():
    """Example: Performance optimization strategies"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Performance Optimization")
    print("="*80)
    
    optimization_strategies = [
        {
            'scenario': 'Limited GPU Memory',
            'solutions': [
                'python pipeline.py --phases compression --use-msmarco --num-examples 25',
                'CUDA_VISIBLE_DEVICES="" python pipeline.py --phases compression --use-msmarco',
                'python pipeline.py --phases compression --use-msmarco --compression-methods rate_based'
            ],
            'description': 'Reduce memory usage for compression'
        },
        {
            'scenario': 'API Rate Limits',
            'solutions': [
                'python pipeline.py --phases generation --input compressed.json --num-examples 50',
                'python pipeline.py --phases evaluation --input responses.json --metrics llm_judge'
            ],
            'description': 'Process in smaller batches to avoid rate limits'
        },
        {
            'scenario': 'Large Dataset Processing',
            'solutions': [
                'python pipeline.py --phases compression --use-msmarco --num-examples 100 --output batch1.json',
                'python pipeline.py --phases generation --input batch1.json --output batch1_responses.json',
                'python pipeline.py --phases evaluation --input batch1_responses.json --output batch1_final.json'
            ],
            'description': 'Process large datasets in manageable batches'
        },
        {
            'scenario': 'Evaluation Only',
            'solutions': [
                'python pipeline.py --phases evaluation --input existing_responses.json',
                'python pipeline.py --phases evaluation --input responses.json --metrics llm_judge --export-msmarco'
            ],
            'description': 'Skip compression and generation for existing responses'
        }
    ]
    
    for strategy in optimization_strategies:
        print(f"\n{strategy['scenario']}:")
        print(f"  {strategy['description']}")
        for solution in strategy['solutions']:
            print(f"  {solution}")


def create_comprehensive_example():
    """Create a comprehensive example that demonstrates all features"""
    print("\n" + "="*80)
    print("COMPREHENSIVE EXAMPLE: Complete Workflow")
    print("="*80)
    
    # Create sample data with responses for evaluation example
    sample_with_responses = [
        {
            "query_id": 1,
            "query": "What is the capital of France?",
            "ground_truth": "Paris is the capital of France.",
            "contexts": ["France is a country in Western Europe. Its capital and largest city is Paris."],
            "is_selected": [1],
            "original": {
                "context": "France is a country in Western Europe. Its capital and largest city is Paris.",
                "response": "The capital of France is Paris.",
                "token_count": 50
            },
            "rate_based": {
                "context": "France... Paris capital",
                "response": "Paris is France's capital.",
                "compression_result": {
                    "compression_rate": "40%",
                    "compressed_tokens": 20
                }
            }
        }
    ]
    
    with open('comprehensive_example.json', 'w') as f:
        json.dump(sample_with_responses, f, indent=2)
    
    print("✓ Created comprehensive_example.json")
    
    comprehensive_workflow = [
        {
            'phase': '1. MS-MARCO Compression',
            'command': 'python pipeline.py --phases compression --use-msmarco --num-examples 10 --output msmarco_compressed.json --verbose',
            'description': 'Compress MS-MARCO dataset with verbose logging'
        },
        {
            'phase': '2. Response Generation',
            'command': 'python pipeline.py --phases generation --input msmarco_compressed.json --output msmarco_responses.json',
            'description': 'Generate responses for compressed data'
        },
        {
            'phase': '3. Comprehensive Evaluation',
            'command': 'python pipeline.py --phases evaluation --input msmarco_responses.json --output msmarco_final.json --metrics llm_judge bleu rouge --export-msmarco --verbose',
            'description': 'Evaluate with all metrics and export MS-MARCO format'
        },
        {
            'phase': '4. Custom Data Processing',
            'command': 'python pipeline.py --phases all --input comprehensive_example.json --output custom_final.json --verbose',
            'description': 'Process custom data through all phases'
        }
    ]
    
    for step in comprehensive_workflow:
        print(f"\n{step['phase']}:")
        print(f"  Description: {step['description']}")
        print(f"  Command: {step['command']}")
    
    print(f"\nThis comprehensive example demonstrates:")
    print(f"  • MS-MARCO dataset integration")
    print(f"  • Custom data processing")
    print(f"  • All compression methods")
    print(f"  • Response generation")
    print(f"  • Multi-metric evaluation")
    print(f"  • MS-MARCO format export")
    print(f"  • Verbose logging throughout")


def main():
    """Run all examples and create demonstration files"""
    print("LLMLingua2 Pipeline - Comprehensive Usage Examples")
    print("=" * 80)
    print("This script demonstrates various ways to use the LLMLingua2 pipeline")
    print("with both MS-MARCO dataset and custom data.")
    print()
    
    # Check if required files exist
    required_files = ['pipeline.py', 'compressor.py', 'generator.py', 'evaluator.py', 'config.json']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"⚠️  Warning: Missing files: {missing_files}")
        print("Make sure all pipeline files are in the current directory.")
        print()
    
    # Run all examples
    example_msmarco_usage()
    example_custom_data_usage()
    example_advanced_configurations()
    example_batch_processing()
    example_evaluation_scenarios()
    example_logging_and_debugging()
    example_performance_optimization()
    create_comprehensive_example()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Update config.json with your ScaleDown API key")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Try MS-MARCO: python pipeline.py --phases compression --use-msmarco --num-examples 5")
    print("4. Try custom data: python pipeline.py --phases all --input sample_data.json --verbose")
    print("5. Check the README.md for detailed documentation")
    
    print(f"\nFiles created in current directory:")
    created_files = [
        'sample_data.json', 'comprehensive_example.json',
        'basic_config.json', 'advanced_config.json', 'msmarco_focused_config.json'
    ]
    for f in created_files:
        if Path(f).exists():
            print(f"  ✓ {f}")
    
    print(f"\nFor more information, see:")
    print(f"  • README.md - Complete documentation")
    print(f"  • config.json - Configuration reference")
    print(f"  • pipeline.py --help - Command line help")


if __name__ == '__main__':
    main()