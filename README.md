# WOPE - AI Poetry Generator

**WOPE** (Word OPErator) is an AI-powered tool that creates unique poems by blending semantic concepts together. It uses advanced language models to generate creative verse with customizable themes and patterns.

## What Does It Do?

WOPE generates poetry by:
- Taking two "semantic words" (themes or concepts) you choose
- Weaving these concepts throughout the poem in wave-like patterns
- Creating multiple verses with controlled rhythm and structure

For example, you might choose "ocean" and "memory" - WOPE will generate a poem that rhythmically alternates between these concepts, creating interesting thematic interplay.

## Quick Start

### Option 1: Run with Docker (Easiest)

If you have Docker installed:

```bash
docker compose up --build
```

Then open your browser to: **http://localhost:8501**

### Option 2: Run Locally

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the web interface:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser to:** **http://localhost:8501**

## How to Use the Web Interface

1. **Enter two semantic words** - Choose any concepts you want in your poem (e.g., "love", "storm", "digital", "nature")

2. **Adjust the sliders:**
   - **Intensity (1-10)**: How strongly each word influences the poem
   - **Period**: How often the semantic waves repeat (smaller = more frequent)

3. **Optional: Add a beginning text** - Start your poem with a custom phrase

4. **Click "Generate Poem"** - Wait a moment while the AI creates your unique poem

5. **Read and enjoy!** - Each generation is unique, so feel free to generate multiple times

## Understanding the Parameters

- **Semantic Word**: The central concept or theme you want woven into the poem
- **Intensity**: Think of this as the "volume" of that theme (higher = more prominent)
- **Period**: Controls the rhythmic pattern - like waves in the ocean, smaller periods create faster oscillations between themes
- **Number of Verses**: How many stanzas in your poem (default: 4)
- **Tokens per Verse**: Roughly the length of each verse (default: 20 words)

## Examples

**Conservative approach:**
- Word 1: "ocean", Intensity: 3, Period: 5
- Word 2: "time", Intensity: 3, Period: 7

**Bold experimentation:**
- Word 1: "chaos", Intensity: 8, Period: 2
- Word 2: "silence", Intensity: 9, Period: 3

## Technical Details

### Architecture

WOPE implements a novel approach to controlled text generation through semantic wave modulation:

**Core Technologies:**
- **GPT-2 Language Model**: Foundation for text generation
- **PyTorch**: Tensor operations and neural network components
- **Semantic Embeddings**: Word2Vec-style similarity using GPT-2's embedding layer
- **Custom Logits Processors**: Real-time manipulation of token probabilities
- **CMU Pronouncing Dictionary**: Vocabulary filtering for pronounceable output

**Key Features:**
- **Lazy Model Loading**: Streamlit caching for fast session management
- **Beam Search Generation**: 10 beams for diverse, high-quality outputs
- **Wave Modulation**: Periodic boosting of semantically similar tokens
- **Stateful Generation**: Sequential verse generation with context preservation
- **Error Handling**: Comprehensive logging and graceful degradation

**Performance:**
- No TensorFlow dependency (pure PyTorch/NumPy)
- Optimized Docker builds with model pre-caching
- Efficient sparse tensor operations
- Configurable through centralized constants

## Research Background

WOPE implements computational creativity concepts from academic research. For the theoretical background, see the [ICCC 2024 paper](https://computationalcreativity.net/iccc24/wp-content/uploads/2023/12/PerezBenavente_ECS_ICCC24.pdf).

## Troubleshooting

**Problem**: Docker container won't start
- **Solution**: Make sure Docker is running and you have sufficient disk space

**Problem**: "Out of memory" errors
- **Solution**: Close other applications or reduce the number of verses

**Problem**: Poems don't reflect my semantic words
- **Solution**: Try increasing the intensity values or decreasing the period

## Contributing

Feel free to open issues or submit pull requests on GitHub!

## License

See LICENSE file for details.
