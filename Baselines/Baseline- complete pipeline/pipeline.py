#!/usr/bin/env python3
"""
LLMLingua2 Pipeline - Main Entry Point
Supports modular execution of compression, generation, and evaluation phases
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Import utils first to get setup_logging
from utils import load_config, setup_logging, validate_input_file

# Set up basic logging before importing other modules
setup_logging(verbose=False)  # Will be reconfigured in main()

from compressor import CompressionPipeline
from generator import ResponseGenerator  
from evaluator import EvaluationPipeline


def run_compression_phase(config: dict, input_file: str, output_file: str, 
                         compression_methods: Optional[List[str]] = None,
                         num_examples: Optional[int] = None,use_msmarco: bool=False) -> str:
    """Run compression phase"""
    print("=" * 60)
    print("PHASE 1: PROMPT COMPRESSION")
    print("=" * 60)
    
    pipeline = CompressionPipeline(config)
    if use_msmarco:
        # Load directly from MS-MARCO
        results = pipeline.compress_from_msmarco(
            compression_methods=compression_methods,
            num_examples=num_examples
        )
    else:
        # Load from file
        results = pipeline.compress_from_file(
            input_file, 
            compression_methods=compression_methods,
            num_examples=num_examples
        )
    pipeline.save_results(results, output_file)
    
    print(f"Compression completed. Results saved to: {output_file}")
    return output_file


def run_generation_phase(config: dict, input_file: str, output_file: str,
                        response_types: Optional[List[str]] = None) -> str:
    """Run response generation phase"""
    print("=" * 60)
    print("PHASE 2: RESPONSE GENERATION") 
    print("=" * 60)
    
    generator = ResponseGenerator(config)
    results = generator.generate_from_file(input_file, response_types=response_types)
    generator.save_results(results, output_file)
    
    print(f"Response generation completed. Results saved to: {output_file}")
    return output_file


def run_evaluation_phase(config: dict, input_file: str, output_file: str,
                        metrics: Optional[List[str]] = None,
                        export_msmarco: bool = False,
                        csv_output: Optional[str] = None) -> str:
    """Run evaluation phase"""
    print("=" * 60)
    print("PHASE 3: EVALUATION")
    print("=" * 60)
    
    evaluator = EvaluationPipeline(config)
    results = evaluator.evaluate_from_file(input_file, metrics=metrics)
    evaluator.save_results(results, output_file)
    
    # Export MS-MARCO format if requested
    if export_msmarco:
        msmarco_dir = Path(output_file).parent / "msmarco_eval"
        evaluator.export_msmarco_format(results, msmarco_dir)
        print(f"MS-MARCO evaluation files exported to: {msmarco_dir}")
    
    # Generate evaluation table and save CSV (use provided path or derive from JSON output)
    derived_csv = Path(output_file).with_suffix('').as_posix() + "_summary.csv"
    evaluator.print_evaluation_table(results, csv_path=csv_output or derived_csv)
    
    # Also export detailed per-example CSV next to summary (unless a custom csv_output was provided to override summary only)
    detailed_csv = Path(output_file).with_suffix('').as_posix() + "_details.csv"
    evaluator.export_detailed_csv(results, detailed_csv, metrics=metrics)
    
    print(f"Evaluation completed. Results saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="LLMLingua2 Pipeline - Modular prompt compression, generation, and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all phases with MS-MARCO dataset
  python pipeline.py --phases all --use-msmarco --config config.json
  
  # Run all phases with custom input file
  python pipeline.py --phases all --input data.json --config config.json
  
  # Run only compression with MS-MARCO
  python pipeline.py --phases compression --use-msmarco --output compressed.json
  
  # Run only compression with custom file
  python pipeline.py --phases compression --input data.json --output compressed.json
  
  # Run generation and evaluation
  python pipeline.py --phases generation evaluation --input compressed.json
  
  # Run with specific methods/metrics
  python pipeline.py --phases compression --use-msmarco --compression-methods rate_based token_based
  python pipeline.py --phases evaluation --metrics llm_judge bleu rouge
        """
    )
    
    # Core arguments
    parser.add_argument('--phases', nargs='+', 
                       choices=['compression', 'generation', 'evaluation', 'all'],
                       required=True,
                       help='Which phases to run')
    parser.add_argument('--input', help='Input JSON file (optional when using MS-MARCO)')
    parser.add_argument('--config', default='config.json', help='Configuration file')
    parser.add_argument('--output', help='Output file (auto-generated if not specified)')
    parser.add_argument('--csv-output', help='CSV summary output path (evaluation phase)')
    
    # Data source options
    parser.add_argument('--use-msmarco', action='store_true',
                       help='Load MS-MARCO dataset directly (for compression phase)')
    

    # Phase-specific arguments
    parser.add_argument('--compression-methods', nargs='*',
                       help='Specific compression methods to use')
    parser.add_argument('--response-types', nargs='*', 
                       help='Specific response types to generate')
    parser.add_argument('--metrics', nargs='*',
                       help='Evaluation metrics to use')
    parser.add_argument('--num-examples', type=int,
                       help='Limit number of examples to process')
    
    # Output options  
    parser.add_argument('--export-msmarco', action='store_true',
                       help='Export MS-MARCO evaluation format')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging with verbose setting
    setup_logging(verbose=args.verbose)
    
    # Ensure all existing loggers use the new level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        # Update all existing loggers to DEBUG level
        for logger_name in ['compressor', 'generator', 'evaluator', 'utils']:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Validate input file (only required if not using MS-MARCO)
    if not args.use_msmarco and args.input and not validate_input_file(args.input):
        print(f"Error: Input file {args.input} not found or invalid")
        sys.exit(1)
    
    # Check if input is required for the phases
    if not args.use_msmarco and not args.input and 'compression' not in phases:
        print("Error: Input file is required for generation and evaluation phases when not using MS-MARCO")
        sys.exit(1)
    
    # Handle 'all' phases
    phases = args.phases
    if 'all' in phases:
        phases = ['compression', 'generation', 'evaluation']
    
    # Process phases sequentially
    current_input = args.input
    
    try:
        for i, phase in enumerate(phases):
            # Auto-generate output filename if not specified
            if args.output:
                if len(phases) == 1:
                    output_file = args.output
                else:
                    # Multi-phase: add phase suffix
                    base = Path(args.output).stem
                    ext = Path(args.output).suffix
                    output_file = f"{base}_{phase}{ext}"
            else:
                # Auto-generate based on input filename or phase
                if current_input:
                    base = Path(current_input).stem
                    output_file = f"{base}_{phase}.json"
                else:
                    # For MS-MARCO loading
                    output_file = f"msmarco_{phase}.json"
            
            
            if phase == 'compression':
                output_file = run_compression_phase(
                    config, current_input, output_file,
                    compression_methods=args.compression_methods,
                    num_examples=args.num_examples,
                    use_msmarco=args.use_msmarco
                )
            elif phase == 'generation':
                output_file = run_generation_phase(
                    config, current_input, output_file,
                    response_types=args.response_types
                )
            elif phase == 'evaluation':
                output_file = run_evaluation_phase(
                    config, current_input, output_file,
                    metrics=args.metrics,
                    export_msmarco=args.export_msmarco,
                    csv_output=args.csv_output
                )
            
            # Use output of current phase as input for next phase
            current_input = output_file
    
    except Exception as e:
        print(f"Pipeline error: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == '__main__':
    main()