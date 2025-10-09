"""
Configuration constants for WOPE poetry generation.

This module centralizes magic numbers and configuration values used throughout
the codebase to improve maintainability and clarity.
"""

# Token IDs
EOS_TOKEN_ID = 50256  # GPT-2 end-of-sequence token <|endoftext|>
PAD_TOKEN_ID = 0      # Padding token ID

# Generation defaults
DEFAULT_NUM_VERSES = 4
DEFAULT_VERSE_SIZE = 20
DEFAULT_NO_REPEAT_NGRAM = 1

# Beam search configuration
NUM_BEAMS = 10
NUM_RETURN_BEAMS = 5

# Semantic similarity
TOP_SIMILAR_TOKENS = 50  # Number of semantically similar tokens to retrieve
DEFAULT_INTENSITY = 6    # Default boost intensity for semantic tokens
DEFAULT_PERIOD = 3       # Default period for semantic wave modulation

# Streamlit UI
MIN_STRENGTH = 1
MAX_STRENGTH = 10
DEFAULT_STRENGTH = 5
