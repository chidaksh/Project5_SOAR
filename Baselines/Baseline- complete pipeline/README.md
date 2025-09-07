# (LLMLingua2) Baseline Pipeline - Prompt Compression and Evaluation

A modular pipeline for prompt compression, response generation, and comprehensive evaluation using LLMLingua2. This system supports both custom datasets and direct MS-MARCO dataset integration with flexible phase execution and logging.

Currently for llmlingua2 compression methods and judge with llm evaluation. Can easily be integrated to othe compression methods by modifying compressor.py and pipeline.py. 

Can easily add other metrics by modifying metrics.py, evaluator.py and pipeline.py.

This folder also has LLMLingua2 train script modified original train script to train using smaller compute and lesser api usage.
## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [MS-MARCO Dataset Support](#ms-marco-dataset-support)
- [Data Formats](#data-formats)
- [Advanced Usage](#advanced-usage)
- [Logging and Debugging](#logging-and-debugging)
- [Performance Optimization](#performance-optimization)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Features

### Core Capabilities
- **Modular Architecture**: Execute compression, generation, or evaluation phases independently
- **MS-MARCO Integration**: Direct loading and processing of MS-MARCO dataset
- **Multiple Compression Methods**: Rate-based, token-based, and context-based compression
- **Comprehensive Evaluation**: LLM-as-a-Judge, BLEU, ROUGE, and MS-MARCO compatibility
- **Professional Logging**: Verbose debugging and detailed progress tracking
- **Flexible Configuration**: JSON-based configuration with validation
- **Resource Optimization**: GPU usage only when required, efficient memory management

### Technical Features
- **Type Safety**: Full type hints and validation
- **Error Handling**: Graceful failure recovery and detailed error reporting
- **Batch Processing**: Support for large dataset processing
- **API Integration**: Seamless integration with ScaleDown API
- **Export Formats**: MS-MARCO evaluation format export

## System Architecture

```
pipeline.py         - Main entry point with phase orchestration
├── compressor.py   - Phase 1: Prompt compression using LLMLingua2
├── generator.py    - Phase 2: Response generation via ScaleDown API
├── evaluator.py    - Phase 3: Multi-metric evaluation pipeline
├── metrics.py      - Evaluation metrics implementations
├── utils.py        - Common utilities, logging, and base classes
└── config.json     - Configuration file with dataset settings
```

### Phase Flow
```
Input Data → Compression → Generation → Evaluation → Results
     ↓            ↓           ↓           ↓
  MS-MARCO    LLMLingua2   ScaleDown   Multiple
  or Custom   (can be any       API        Metrics
            compressor model)
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (for compression phase)
- ScaleDown API key

### Setup

```bash
# Clone the repository
git clone https://github.com/ilatims-b/Project5_SOAR.git
cd Project5_SOAR/Baselines/Baseline\-\ complete\ pipeline

# Install core dependencies
pip install -r requirements.txt

# Install optional evaluation dependencies
pip install nltk rouge-score datasets

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Dependencies

**Core Requirements:**
- `llmlingua` - LLMLingua2 compression library
- `requests` - HTTP client for API calls
- `pandas` - Data manipulation
- `datasets` - For MS-MARCO dataset loading

**Optional Requirements:**
- `nltk` - For BLEU evaluation
- `rouge-score` - For ROUGE evaluation

## Configuration

### Configuration File Structure

```json
{
  "context_separator": "<<<>>>",
  "api_config": {
    "api_key": "your_scaledown_api_key",
    "base_url": "https://api.scaledown.xyz/compress/",
    "model": "gemini-2.5-flash"
  },
  "compression_methods": {
    "rate_based": {"rate": 0.3},
    "token_based": {"target_token": 200},
    "context_based": {"target_context": 3}
  },
  "evaluation": {
    "enabled": true,
    "metric": "llm_judge",
    "parameters": {
      "judge_model": "gemini-2.5-flash",
      "threshold": 0.5
    }
  },
  "dataset_config": {
    "version": "v2.1",
    "query_type": "NUMERIC",
    "max_examples": 10,
    "start": 0
  }
}
```

### Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `context_separator` | string | Separator for combining contexts | `"<<<>>>"` |
| `api_config.api_key` | string | ScaleDown API key | Required |
| `api_config.base_url` | string | ScaleDown API endpoint | Required |
| `api_config.model` | string | Model for generation | Required |
| `compression_methods` | object | Compression method configurations | See below |
| `evaluation.enabled` | boolean | Enable evaluation phase | `true` |
| `evaluation.metric` | string | Primary evaluation metric | `"llm_judge"` |
| `dataset_config.version` | string | MS-MARCO dataset version | `"v2.1"` |
| `dataset_config.query_type` | string | Query type filter | `"NUMERIC"` |
| `dataset_config.max_examples` | integer | Maximum examples to process | `100` |
| `dataset_config.start` | integer | Starting index for examples | `0` |

### Compression Methods

**Rate-based Compression:**
```json
"rate_based": {"rate": 0.3}  // Keep 30% of tokens
```

**Token-based Compression:**
```json
"token_based": {"target_token": 200}  // Compress to 200 tokens
```

**Context-based Compression:**
```json
"context_based": {"target_context": 3}  // Keep 3 most relevant contexts
```

### Long Input Handling

LLMLingua2 chunking is enabled in the compressor to support long inputs (beyond 512 tokens). Inputs are split using the configured `context_separator`, newlines, and sentence punctuation. No manual truncation is applied.

## Quick Start

### 1. Basic Usage with Custom Data

```bash
# Run all phases with custom input
python pipeline.py --phases all --input data.json --config config.json --verbose

# Run individual phases
python pipeline.py --phases compression --input data.json --output compressed.json
python pipeline.py --phases generation --input compressed.json --output responses.json
python pipeline.py --phases evaluation --input responses.json --output final.json
```

### 2. MS-MARCO Dataset Usage

```bash
# Run all phases with MS-MARCO dataset
python pipeline.py --phases all --use-msmarco --config config.json --verbose

# Run compression only with MS-MARCO
python pipeline.py --phases compression --use-msmarco --output compressed.json

# Run with specific compression methods
python pipeline.py --phases compression --use-msmarco --compression-methods rate_based token_based
```

### 3. Evaluation with Specific Metrics

```bash
# Run evaluation with specific metrics
python pipeline.py --phases evaluation --input responses.json --metrics llm_judge bleu rouge

# Export MS-MARCO format for external evaluation
python pipeline.py --phases evaluation --input responses.json --export-msmarco
```

The evaluation phase also emits CSV files (see Outputs section below).

## MS-MARCO Dataset Support

### Automatic Dataset Loading

The system automatically loads MS-MARCO dataset when `--use-msmarco` flag is used:

```bash
# Load MS-MARCO with default settings (100 examples, NUMERIC queries)
python pipeline.py --phases compression --use-msmarco

# Load with custom parameters
python pipeline.py --phases compression --use-msmarco --num-examples 50
```

### Dataset Configuration

Configure MS-MARCO loading in `config.json`:

```json
{
  "dataset_config": {
    "version": "v2.1",           // MS-MARCO version
    "query_type": "NUMERIC",     // Filter by query type
    "max_examples": 100,         // Maximum examples to process
    "start": 0                   // Starting index
  }
}
```

### Supported Query Types

- `"NUMERIC"` - Numeric answer questions
- `"BOOLEAN"` - Yes/No questions
- `"DATE"` - Date-related questions
- `"ENTITY"` - Entity-based questions

## Outputs (JSON and CSV)

The evaluation phase writes the following alongside the JSON results:

- Summary CSV (aggregated by response type): `<output>_summary.csv` (or a custom path via `--csv-output`). Includes metrics per response type, `Avg Tokens`, `Compression Rate (%)`, and `Compression Ratio (x)`.
- Detailed CSV (per example, all response types including `original`): `<output>_details.csv`. Columns: `query_id`, `query`, `response_type`, `tokens`, `compression_ratio`, `compression_rate`, plus metric columns (e.g., `llm_judge`, `bleu`, `rouge`).

## Data Formats

### Input Format (Custom Data)

```json
[
  {
    "query_id": 123,
    "query": "What is the average temperature?",
    "ground_truth": "The average global temperature is 14°C.",
    "contexts": [
      "Context 1 text here...",
      "Context 2 text here..."
    ],
    "is_selected": [0, 1]
  }
]
```

### Phase 1 Output (After Compression)

```json
{
  "query_id": 123,
  "query": "What is the average temperature?",
  "ground_truth": "The average global temperature is 14°C.",
  "contexts": ["Context 1 text here...", "Context 2 text here..."],
  "is_selected": [0, 1],
  "original": {
    "context": "combined contexts",
    "token_count": 500
  },
  "rate_based": {
    "context": "compressed context",
    "compression_result": {
      "compression_rate": "30%",
      "compression_ratio": "3.3x",
      "original_tokens": 500,
      "compressed_tokens": 150,
      "context_analysis": {
        "0": {"original_length": 100, "retained_count": 30, "retention_ratio": 0.3},
        "1": {"original_length": 400, "retained_count": 120, "retention_ratio": 0.3}
      }
    }
  }
}
```

### Phase 2 Output (After Response Generation)

```json
{
  // ... previous data
  "original": {
    "context": "combined contexts",
    "token_count": 500,
    "response": "The average global temperature is approximately 14°C."
  },
  "rate_based": {
    "context": "compressed context",
    "compression_result": {...},
    "response": "The average global temperature is 14°C."
  }
}
```

### Phase 3 Output (After Evaluation)

```json
{
  // ... previous data
  "original": {
    "context": "combined contexts",
    "token_count": 500,
    "response": "The average global temperature is approximately 14°C.",
    "evaluations": {
      "llm_judge": 1,
      "bleu": 0.85,
      "rouge": 0.92
    }
  },
  "rate_based": {
    "context": "compressed context",
    "compression_result": {...},
    "response": "The average global temperature is 14°C.",
    "evaluations": {
      "llm_judge": 0.8,
      "bleu": 0.75,
      "rouge": 0.88
    }
  }
}
```

## Advanced Usage

### Command Line Options

```bash
# Core options
--phases {compression,generation,evaluation,all}  # Phases to run
--input INPUT_FILE                                # Input JSON file (optional with MS-MARCO)
--config CONFIG_FILE                              # Configuration file
--output OUTPUT_FILE                              # Output file (auto-generated if not specified)
--use-msmarco                                     # Use MS-MARCO dataset instead of input file

# Phase-specific options
--compression-methods METHOD [METHOD ...]         # Specific compression methods
--response-types TYPE [TYPE ...]                  # Specific response types
--metrics METRIC [METRIC ...]                     # Evaluation metrics
--num-examples N                                  # Limit number of examples

# Output options
--export-msmarco                                  # Export MS-MARCO evaluation format
--verbose                                         # Enable verbose logging
```

### Batch Processing

```bash
# Process large datasets in batches
python pipeline.py --phases compression --use-msmarco --num-examples 100 --output batch1.json
python pipeline.py --phases compression --use-msmarco --num-examples 100 --output batch2.json

# Combine results for evaluation
python pipeline.py --phases generation --input batch1.json --output batch1_responses.json
python pipeline.py --phases evaluation --input batch1_responses.json --output batch1_final.json
```

### Custom Evaluation Metrics

Add custom metrics to `metrics.py`:

```python
class CustomEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def evaluate(self, ground_truth: str, generated_response: str) -> float:
        # Your evaluation logic here
        return score
```

## Logging and Debugging

### Logging Levels

- **INFO**: General progress information
- **DEBUG**: Detailed debugging information (use `--verbose`)
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors

### Verbose Logging

Enable detailed logging with the `--verbose` flag:

```bash
# Enable verbose logging for all phases
python pipeline.py --phases all --use-msmarco --verbose

# Verbose logging shows:
# - Detailed compression progress
# - API request/response details
# - Evaluation metric calculations
# - Context retention analysis
```

### Log Output Format

```
2025-09-07 13:16:31 - compressor - INFO - Initializing LLMLingua2 compressor...
2025-09-07 13:16:31 - compressor - DEBUG - Loading model: microsoft/llmlingua-2-xlm-roberta-large-meetingbank
2025-09-07 13:16:31 - compressor - INFO - Processing example 1/100: What is the capital of France?...
2025-09-07 13:16:31 - compressor - DEBUG - Compressing with rate_based method...
```

## Performance Optimization

### Resource Requirements by Phase

| Phase | GPU Required | Memory | Network |
|-------|-------------|--------|---------|
| Compression | Yes | ~2GB VRAM | Model download only |
| Generation | No | ~500MB RAM | API calls |
| Evaluation | No | ~500MB RAM | API calls |

### Recommended Workflow

1. **GPU Machine**: Run compression phase only
   ```bash
   python pipeline.py --phases compression --use-msmarco --output compressed.json
   ```

2. **CPU Machine**: Run generation and evaluation
   ```bash
   python pipeline.py --phases generation --input compressed.json --output responses.json
   python pipeline.py --phases evaluation --input responses.json --output final.json
   ```

### Memory Optimization

```bash
# Process in smaller batches
python pipeline.py --phases compression --use-msmarco --num-examples 50

# Use CPU for compression (slower but less memory)
CUDA_VISIBLE_DEVICES="" python pipeline.py --phases compression --use-msmarco
```

## API Reference

### CompressionPipeline

```python
class CompressionPipeline(BaseComponent):
    def __init__(self, config: Dict[str, Any])
    def compress_from_file(self, input_file: str, compression_methods: Optional[List[str]] = None, num_examples: Optional[int] = None) -> List[Dict]
    def compress_from_msmarco(self, compression_methods: Optional[List[str]] = None, num_examples: Optional[int] = None) -> List[Dict]
```

### ResponseGenerator

```python
class ResponseGenerator(BaseComponent):
    def __init__(self, config: Dict[str, Any])
    def generate_from_file(self, input_file: str, response_types: Optional[List[str]] = None) -> List[Dict]
```

### EvaluationPipeline

```python
class EvaluationPipeline(BaseComponent):
    def __init__(self, config: Dict[str, Any])
    def evaluate_from_file(self, input_file: str, metrics: Optional[List[str]] = None) -> List[Dict]
    def export_msmarco_format(self, results: List[Dict], output_dir: Path)
```

## Troubleshooting

### Common Issues

**CUDA Memory Issues:**
```bash
# Reduce batch size
python pipeline.py --phases compression --use-msmarco --num-examples 25

# Use CPU instead
CUDA_VISIBLE_DEVICES="" python pipeline.py --phases compression --use-msmarco
```

**API Rate Limits:**
- Add delays between requests in the code
- Use different models for generation vs evaluation
- Process in smaller batches

**Missing Dependencies:**
```bash
# Install missing packages
pip install nltk rouge-score datasets

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**MS-MARCO Loading Issues:**
```bash
# Check dataset availability
python -c "from datasets import load_dataset; print(load_dataset('microsoft/ms_marco', 'v2.1'))"
```

### Debug Mode

Enable debug logging to diagnose issues:

```bash
python pipeline.py --phases all --use-msmarco --verbose --num-examples 1
```

## Contributing

### Adding New Features

1. **New Compression Methods**: Add to `compressor.py` and update config schema
2. **New Evaluation Metrics**: Add to `metrics.py` and update evaluator
3. **New Data Sources**: Extend `utils.py` data loading functions


---

For more examples and detailed usage, see `examples.py` in the repository.
