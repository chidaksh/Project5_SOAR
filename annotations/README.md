# Passage Attribution Analysis - Log-Probability Method

## Overview

This implementation measures which passages in MS-MARCO contribute to the model's ability to generate correct answers using **log-probability attribution**.

## Methodology

### Core Metric

We measure `log P(answer | context, question)` directly by:
1. Concatenating prompt + gold answer
2. Computing per-token cross-entropy loss only on answer tokens
3. Masking prompt and padding tokens with `-100` (ignore_index)
4. Summing negative log-likelihood to get log-probability

### Attribution Calculation

For each sample:
- **Baseline**: `log_prob(answer | full_context, question)`
- **No-context**: `log_prob(answer | "", question)`
- **Ablation_i**: `log_prob(answer | context_without_passage_i, question)`

**Delta**: `Δᵢ = baseline - ablation_i`

**Interpretation**:
- **Positive Δ**: Removing passage decreases confidence → passage is **helpful**
- **Negative Δ**: Removing passage increases confidence → passage is **confusing**
- **Δ ≈ 0**: Passage has no impact

## Files

- **`paragraph_level_analysis.py`**: Main analysis script (500 samples)
- **`test_optimization.py`**: Quick test (3 samples) for validation
- **`requirements.txt`**: Python dependencies

## Key Fixes Applied

### Bug 1: Incorrect Prompt Length Calculation (CRITICAL)
**Problem**: Used padded batch length instead of individual prompt lengths
```python
# ❌ WRONG
prompt_lengths = prompt_inputs.input_ids.shape[1]  # Same for all!

# ✅ CORRECT
prompt_inputs = tokenizer(prompts, padding=False, ...)
prompt_lengths = [len(ids) for ids in prompt_inputs.input_ids]
```

### Bug 2: Padding Tokens Not Masked (CRITICAL)
**Problem**: Loss computed on PAD tokens, inflating negative log-likelihood
```python
# ✅ FIXED - Added padding mask
labels[labels == tokenizer.pad_token_id] = -100
```

### Bug 3: Redundant Code
**Fixed**: Removed unused imports and redundant loss calculations

## Usage

### Test First (Recommended)
```bash
python test_optimization.py
```

**Expected output**:
```
Sample 0: 9 passages
  Baseline log-prob: -8.78
  No-context log-prob: -26664.34
  Delta total: 26655.56
  Selected passage delta: 197.64 (positive = helpful)
  Non-selected mean delta: 81.59
```

✓ Baseline should be LESS negative than no-context  
✓ Selected passages should have HIGHER mean delta  
✓ All log-probs should be DIFFERENT

### Full Analysis
```bash
# Remove any old checkpoints
rm passage_attribution_checkpoint.json 2>/dev/null

# Run full analysis
python paragraph_level_analysis.py
```

**Runtime**: ~6-7 minutes for 500 samples  
**Output**: `passage_attribution_results_logprob.json`

### Resume from Checkpoint
If interrupted, the script automatically resumes from the last checkpoint (saved every 50 samples).

## Expected Results

### Good Attribution Example
```json
{
  "query_id": "6260",
  "question": "average cost of dying",
  "baseline_log_prob": -8.78,
  "no_context_log_prob": -26664.34,
  "delta_total": 26655.56,
  "passage_analysis": [
    {
      "passage_idx": 8,
      "delta_log_prob": 197.64,  // Highest delta
      "is_selected": 1            // Contains the answer
    },
    {
      "passage_idx": 0,
      "delta_log_prob": 45.23,
      "is_selected": 0
    }
    // ... other passages with lower deltas
  ]
}
```

### Statistical Analysis
After processing all samples, the script reports:
- **Mean delta**: Average contribution across all passages
- **Std delta**: Variability in contributions
- **2-sigma threshold**: `mean + 2*std` for identifying highly influential passages

## Configuration

In `paragraph_level_analysis.py`:

```python
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
MAX_CONTEXT_LENGTH = 4000
DATASET_NAME = "ms_marco"
DATASET_CONFIG = "v1.1"
QUERY_TYPES = ["NUMERIC"]
NUM_SAMPLES = 500
BATCH_SIZE = 32              # Adjust based on GPU memory
CHECKPOINT_FREQ = 50         # Save every N samples
```

## Performance Optimizations

1. **Batching**: All prompts (baseline + no-context + ablations) processed together
2. **Flash Attention 2**: Used if available for faster inference
3. **Checkpointing**: Automatic resume capability
4. **Memory Management**: Explicit cleanup every 10 samples

## Troubleshooting

### All deltas are 0.0
- ❌ Bug: Padding tokens not masked
- ✅ Ensure line 70: `labels[labels == tokenizer.pad_token_id] = -100`

### No-context values extremely negative
- ❌ Bug: Individual prompt lengths not used
- ✅ Ensure lines 60-61 use `padding=False` and list comprehension

### Out of memory
- Reduce `BATCH_SIZE` from 32 to 16 or 8
- Reduce `MAX_CONTEXT_LENGTH` if contexts are very long

## Verification

Run verification to ensure fixes are working:
```bash
python test_optimization.py
```

Check that:
1. Log-probs vary across conditions (not all identical)
2. Baseline > no-context (context helps)
3. Selected passages have positive mean delta
4. No warnings about tensor shape mismatches

## Citation

This implementation is based on:
- **Methodology**: Standard log-probability attribution
- **Optimization**: Batched inference for efficiency
- **Dataset**: MS-MARCO v1.1 (numeric queries)
- **Model**: Microsoft Phi-3-mini-4k-instruct
