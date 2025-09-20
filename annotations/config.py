import random
import numpy as np
import torch

RANDOM_SEED = 42



BOTTOM_K_PERCENT_FOR_NULL = 20 # The bottom 20% of tokens will form the null distribution


QUERY_TYPES = ["NUMERIC"]
TOTAL_EXAMPLES = 3
NULL_EXAMPLES = 3

MODEL_ID = "microsoft/Phi-3-mini-128k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONTEXT_LENGTH = 10000

NUM_STOCHASTIC_REPEATS = 3
BATCH_SIZE = 32  # Token batch size for GPU optimization

USE_POSITIONAL_BINNING = True

Q_CRITICAL = 0.15
Q_HARMFUL = 0.05

PUNCTUATION_TOKENS = set('.,;:?!-"\'()[]{}…—–''""')

NULL_STATS_FILE = "null_distribution_stats.json"
RESULTS_FILE = "msmarco_phase3_results.json"
ATTRIBUTIONS_FILE = "token_attributions_detailed.json"

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_config():
    valid_query_types = {"NUMERIC", "ENTITY", "DESCRIPTION", "PERSON", "LOCATION"}
    for qt in QUERY_TYPES:
        if qt not in valid_query_types:
            raise ValueError(f"Invalid query_type: {qt}. Must be one of {valid_query_types}")
    if TOTAL_EXAMPLES < len(QUERY_TYPES):
        raise ValueError(f"TOTAL_EXAMPLES ({TOTAL_EXAMPLES}) must be >= number of query types ({len(QUERY_TYPES)})")

if __name__ == "__main__":
    validate_config()
