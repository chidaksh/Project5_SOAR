# SelectiveContext Evaluation Pipeline

This directory implements a three-phase evaluation pipeline for prompt compression using the Selective Context method on MS-MARCO dataset. The implementation is adapted from [liyucheng09/Selective_Context](https://github.com/liyucheng09/Selective_Context).

## Overview

The evaluation follows a **three-phase pipeline**:
1. **Phase 1: Compress** - Apply SelectiveContext compression to MS-MARCO dataset
2. **Phase 2: Generate** - Generate responses using compressed contexts  
3. **Phase 3: Evaluate** - Assess quality using multiple evaluation metrics

Each phase transforms the dataset by adding new fields while preserving original data for analysis.

## Configuration

All experiments are configured through `experiment.yaml`:

```yaml
# Dataset and output settings
save_dir: "results"
dataset:
  name: "microsoft/ms_marco"
  version: "v2.1"
  split: "validation" 
  query_type: "NUMERIC"        # Filter by query type
  max_examples: 2              # Limit dataset size

# Compression configuration
compressor:
  model_type: "meta-llama/Llama-3.2-3B"
  reduce_ratio: 0.35           # Target compression ratio
  reduce_level: "phrase"       # Compression granularity

# Generation configuration  
generation:
  generator_type: "scaledown"
  api:
    base_url: "https://api.scaledown.xyz/compress/"
    api_key: "your-api-key"
    model: "gemini-2.5-flash"

# Evaluation configuration
evaluation:
  metrics: ["bleu", "rouge", "msmarco"]  # Available: bleu, rouge, msmarco, llm_judge
```

## Phase 1: Compression (`compress.py`)

Applies SelectiveContext compression to MS-MARCO passages.

**Usage:**
```bash
python compress.py [config.yaml]  # Uses experiment.yaml by default
```

**Dataset Structure After Compression:**

Original MS-MARCO fields are preserved, plus new compression-related fields:

```python
{
  # Original MS-MARCO fields
  "query_id": int,
  "query_type": str, 
  "query": str,
  "answers": List[str],
  "wellFormedAnswers": List[str],
  
  # Compressed data
  "passages": {
    "passage_text": List[str],          # Compressed passage texts
    "is_selected": List[bool],          # Original selection flags  
    "url": List[str],                   # Original URLs
    "context_unit": List[List[str]],    # Original context units per passage
    "kept_context_unit_mask": List[List[bool]]  # Compression masks per passage
  },
  
  # Compression metadata
  "original_passages": dict,            # Original uncompressed passages
  "compression_words_ratio": float,     # Words ratio (compressed/original)
  "compression_characters_ratio": float, # Character ratio (compressed/original) 
  "compression_rate": float,            # Target compression rate from config
  "compression_level": str,             # Compression granularity from config
  "compression_error": str              # Error message if compression failed
}
```

**Output:**
- Compressed dataset saved to `results/MS-MARCO-compressed-{ratio}-{level}/`
- Metadata saved to `results/MS-MARCO-compressed-{ratio}-{level}.json`

## Phase 2: Response Generation (`generate.py`)

Generates responses using compressed contexts via configured API.

**Usage:**
```bash  
python generate.py [config.yaml]  # Uses experiment.yaml by default
```

**Dataset Structure After Generation:**

All compression fields are preserved, plus new generation fields:

```python
{
  # ... (all fields from Phase 1) ...
  
  # Generated response data
  "generated_response": str,            # Generated answer text
  "generation_metadata": {
    "generator": str,                   # Generator type (e.g., "scaledown") 
    "model": str,                       # Model name used for generation
    "timestamp": str,                   # ISO format generation timestamp
    "api_response_time": float,         # Response time in seconds (if available)
    "generation_error": str             # Error message if generation failed
  }
}
```

**Features:**
- Incremental processing: Only processes samples without `generated_response`
- Progress saving: Saves intermediate results every 10 samples
- Error handling: Failed generations logged with error messages

## Phase 3: Evaluation (`evaluation.py`)

Evaluates generated responses using multiple metrics against ground truth answers.

**Usage:**
```bash
python evaluation.py [config.yaml]  # Uses experiment.yaml by default  
```

**Available Evaluation Metrics:**
- **BLEU**: N-gram overlap similarity with smoothing
- **ROUGE**: Recall-oriented understudy for gisting evaluation (ROUGE-1, ROUGE-2, ROUGE-L)
- **MS-MARCO**: Official MS-MARCO evaluator (BLEU + ROUGE + F1 + Semantic Similarity)
- **LLM Judge**: LLM-based quality assessment (binary match/no-match)

**Evaluation Results Structure:**

```python
{
  "evaluation_results": [
    {
      "query_id": int,
      "query": str, 
      "ground_truth": str,              # Joined answers
      "generated_response": str,
      
      # Metric scores (depending on configured metrics)
      "bleu": float,                    # BLEU score (0-1)
      "rouge_1": float,                 # ROUGE-1 F-score (0-1)
      "rouge_2": float,                 # ROUGE-2 F-score (0-1) 
      "rouge_l": float,                 # ROUGE-L F-score (0-1)
      "llm_judge_score": int,           # LLM judge score (0 or 1)
      # ... (additional MS-MARCO metrics if enabled)
    }
  ],
  "aggregate_metrics": {
    "avg_bleu": float,                  # Average BLEU across samples
    "avg_rouge_1": float,               # Average ROUGE-1 across samples
    "count_bleu": int,                  # Number of samples with BLEU scores
    # ... (averages and counts for all metrics)
  },
  "evaluation_metadata": {
    "timestamp": str,                   # Evaluation timestamp
    "total_samples": int,               # Number of evaluated samples
    "metrics_used": List[str]           # List of evaluation metrics applied
  }
}
```

**Output:**
- Results saved to `results/MS-MARCO-compressed-{ratio}-{level}_evaluation_results.json`

## Complete Workflow

Run all three phases sequentially:

```bash
# Phase 1: Compress dataset
python compress.py

# Phase 2: Generate responses  
python generate.py

# Phase 3: Evaluate responses
python evaluation.py
```

**File Progression:**
1. `compress.py` → `results/MS-MARCO-compressed-0_35-phrase/`
2. `generate.py` → Updates same dataset with `generated_response` fields
3. `evaluation.py` → `results/MS-MARCO-compressed-0_35-phrase_evaluation_results.json`

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # For MS-MARCO evaluator
python -m spacy download en_core_web_lg  # For MS-MARCO evaluator
```

## Module Structure

- `compress.py` - Main compression script
- `generate.py` - Response generation script  
- `evaluation.py` - Evaluation script
- `compressor/` - SelectiveContext implementation
- `generator/` - Response generation backends
- `evaluator/` - Evaluation metrics implementation
- `utils.py` - Shared utilities (config loading)
- `experiment.yaml` - Experiment configuration