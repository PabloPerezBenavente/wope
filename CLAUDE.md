# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application

```bash
# Run the Streamlit web app (primary interface)
streamlit run streamlit_app.py

# Access at http://localhost:8501
```

### Docker Deployment

```bash
# Build and run with Docker (note: use 'docker compose' without dash)
docker compose up --build

# Stop the container
docker compose down

# Access at http://localhost:8501
```

### Installing Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# All dependencies are now in requirements.txt
pip install -r requirements.txt
```

## Architecture Overview

WOPE is a computational poetry generation system that creates poems by combining semantic concepts through controlled text generation using GPT-2 with custom logits processors.

### Core Components & Data Flow

1. **Copoet (Wope.py)**: Thin wrapper around PoemAgent - the main entry point for programmatic usage

2. **PoemAgent (poem_agent.py)**: Orchestrates the entire generation pipeline
   - Loads GPT-2 model, tokenizer, and vocabulary on initialization
   - Validates vocabulary against CMU pronouncing dictionary (pronounceable words only)
   - Manages `active_tools` dict containing all generation rules and parameters
   - Methods: `introduce_rule()`, `create_input()`, `generate_text()`
   - Vocabulary files loaded from `data/vocab/` relative to the script location

3. **Generator Pipeline (poem_generator.py)**: Implements verse-by-verse beam search generation
   - `get_tool_tools()`: Converts rule dict into LogitsProcessorList and StoppingCriteriaList
   - `generator()`: Main loop that generates each verse sequentially
   - Uses beam search (10 beams, returns top 5) for each verse
   - Handles verse size enforcement via MinNewTokensLengthLogitsProcessor and ForcedEOSTokenLogitsProcessor

4. **MasterLogits (masterlogits.py)**: Custom LogitsProcessor for semantic wave modulation
   - Applies periodic boosting of semantically similar tokens based on wave parameters
   - Boosts tokens at timesteps: `(current_length - first_prediction) % period == 0`
   - Bans semantic tokens on non-boost timesteps to create wave effect
   - Delegates to `boost_scores()` and `ban_scores()` in utils.py

5. **Utils (utils.py)**: Low-level embedding and score manipulation
   - `get_embedding()`: Extracts GPT-2 embeddings for words
   - `get_similarity()`: Calculates cosine similarity between word embeddings
   - `get_semantic_items()`: Finds top 50 semantically similar tokens (filtered by CMU dictionary)
   - `boost_scores()`: Uses NumPy arrays and PyTorch tensors to add intensity values to specific token logits
   - `ban_scores()`: Uses PyTorch sparse tensors to mask tokens with -inf
   - **NOTE**: Pure PyTorch/NumPy implementation for efficient tensor operations

6. **Streamlit Interface (streamlit_app.py)**: Web UI for interactive poem generation
   - Takes two semantic words with intensity/period sliders
   - Allows custom beginning text input
   - Default: 4 verses, 20 tokens per verse

### Key Generation Parameters

When using `introduce_rule()`:
- **cos_sim**: `('cos_sim', word, intensity, period)` - Semantic wave modulation
  - `word`: Must exist in GPT-2 vocab and CMU dictionary
  - `intensity`: 1-10, strength of boost applied to similar tokens
  - `period`: Frequency of semantic wave (boost every N tokens)
- **verse_size**: `('verse_size', num_tokens)` - Fixed tokens per verse (default: 20)
- **num_verses**: `('num_verses', count)` - Number of verses to generate (default: 4)
- **no_repeat**: `('no_repeat', n)` - Prevent n-gram repetition (default: 1)

### Important Implementation Details

1. **Path Handling**: Vocabulary files are loaded using `os.path` relative to the script location (poem_agent.py:36-42), ensuring compatibility with both local development and Docker environments.

2. **Tensor Device Management**: When changing semantic words, tensors must be moved to CPU (utils.py:40, 43) to avoid device mismatch errors.

3. **Generation Loop**: Each verse is generated independently but continues from the previous verse's output. The `input_length` parameter tracks cumulative token count across verses.

4. **EOS Token Handling**: Token ID 50256 (`<|endoftext|>`) is used as verse separator but removed from intermediate prompts to allow continuation.

5. **Vocabulary Validation**: Only tokens that exist in CMU pronouncing dictionary are used for semantic similarity to ensure pronounceable poetic output.

## Research Reference

This project implements concepts from computational creativity research. See the [ICCC 2024 paper](https://computationalcreativity.net/iccc24/wp-content/uploads/2023/12/PerezBenavente_ECS_ICCC24.pdf) for theoretical background.
