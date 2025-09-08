# WOPE 🎭 - Word-Oriented Poetry Engine

A computational poetry generation system that creates unique poems by blending semantic concepts through controlled text generation using GPT-2.

## Overview

WOPE generates poetic text by combining two different semantic domains or concepts, creating unexpected and creative combinations that can serve as inspiration for poetry. The system uses wave-based semantic modulation to periodically influence word selection, resulting in text that oscillates between different conceptual realms.

## Features

- **Dual Semantic Control**: Blend two different words or concepts with adjustable prominence
- **Wave-Based Modulation**: Semantic influences that ebb and flow throughout the generated text
- **Structured Generation**: Configurable verse length and count
- **Interactive Web Interface**: User-friendly Streamlit application
- **Repetition Prevention**: Automatic n-gram filtering to avoid repetitive patterns

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/wope.git
cd wope
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional required packages:
```bash
pip install torch transformers tensorflow numpy
```

## Usage

### Web Interface

Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

Then open your browser and navigate to `http://localhost:8501`

### Using the Web Interface

1. **Enter two words**: These will be the semantic anchors for your poem
2. **Adjust prominence sliders**: Control how strongly each word influences the generation (1-10)
3. **Provide a beginning**: Start your poem with a phrase or sentence
4. **Generate**: Click the button to create your poem

### Programmatic Usage

```python
from Wope import Copoet

# Initialize the poetry agent
poet = Copoet(num_verses=4, no_repeat=1)

# Set the input text
poet.create_input("Once upon a")

# Add semantic rules
poet.introduce_rule(('cos_sim', 'dream', 7, 5))  # word, intensity, period
poet.introduce_rule(('cos_sim', 'machine', 10, 3))

# Generate poem
poem = poet.generate_text()
print(poem)
```

## How It Works

WOPE uses a modified GPT-2 language model with custom logits processors to guide text generation:

1. **Semantic Embedding**: Words are converted to high-dimensional vectors using GPT-2's embeddings
2. **Cosine Similarity**: The system finds tokens semantically similar to your chosen words
3. **Wave Modulation**: Semantic influences are applied in waves, creating rhythmic conceptual shifts
4. **Controlled Generation**: Custom stopping criteria and processors ensure structured output

## Configuration Options

- **num_verses**: Number of verses to generate (default: 4)
- **verse_size**: Number of tokens per verse (default: 20)
- **no_repeat**: N-gram size for repetition prevention (default: 1)
- **intensity**: Strength of semantic influence (1-10)
- **period**: Wavelength of semantic modulation

## Project Structure

```
wope/
├── streamlit_app.py      # Web interface
├── Wope.py              # Main Copoet class
├── poem_agent.py        # Core poetry generation agent
├── poem_generator.py    # Generation pipeline and tools
├── masterlogits.py      # Custom logits processor
├── endcriteria.py       # Stopping criteria
├── utils.py             # Embedding and similarity utilities
└── data/
    └── vocab/           # GPT-2 vocabulary files
```

## Research

This project implements concepts from computational creativity research. For more details about the theoretical background and methodology, see the [research paper](https://computationalcreativity.net/iccc24/wp-content/uploads/2023/12/PerezBenavente_ECS_ICCC24.pdf).

## Limitations

- Complex or rare words may not be available in the vocabulary
- Generation quality depends on the semantic relationship between chosen words
- The system requires both PyTorch and TensorFlow due to mixed dependencies

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source. Please check the repository for license details.

## Acknowledgments

- Built with [Hugging Face Transformers](https://huggingface.co/transformers/)
- Uses GPT-2 model by OpenAI
- CMU Pronouncing Dictionary for vocabulary filtering