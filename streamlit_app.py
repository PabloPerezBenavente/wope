import streamlit as st
from Wope import Copoet
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for wave aesthetics and gradient background
st.markdown("""
<style>
    /* Gradient background with wave theme */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradientWave 15s ease infinite;
    }

    @keyframes gradientWave {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Main content container */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }

    /* Title styling with wave effect */
    h1 {
        background: linear-gradient(120deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Subtitle and description */
    .stMarkdown p {
        color: #000000;
        font-size: 1.1rem;
        line-height: 1.6;
    }

    /* Input fields with wave accent */
    .stTextInput > div > div > input {
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: white;
    }

    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Slider customization with wave colors */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    /* Button styling with wave theme */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 32px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* Poem display card */
    .poem-container {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(240, 147, 251, 0.05));
        border-left: 4px solid #667eea;
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }

    /* Subheaders with wave accent */
    h2, h3 {
        color: #000000;
        font-weight: 600;
    }

    /* Wave divider */
    .wave-divider {
        height: 4px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        margin: 2rem 0;
        border-radius: 2px;
    }

    /* Info box styling */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }

    /* Label styling */
    .stTextInput label, .stSlider label {
        color: #000000;
        font-weight: 600;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the model once and cache it for the session."""
    logger.info("Loading GPT-2 model...")
    try:
        return Copoet()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error("Failed to initialize the poetry generator. Please refresh the page.")
        st.stop()

def generate_poem_stream(word1=None,
                         strength1=None,
                         word2=None,
                         strength2=None,
                         beginning=None):
    """
    Generates a poem using two semantic words, streaming verses as they're created.

    Args:
        word1 (str): First semantic word.
        strength1 (int): Strength/prominence of first word (1-10).
        word2 (str): Second semantic word.
        strength2 (int): Strength/prominence of second word (1-10).
        beginning (str): Initial text to start the poem.

    Yields:
        tuple: (verse_num, verse_text) or ('error', error_message)
    """
    try:
        ai_poet = load_model()
        ai_poet.create_input(beginning)
        period_1 = 11 - strength1
        period_2 = 11 - strength2
        ai_poet.introduce_rule(('num_verses', 4))

        try:
            ai_poet.introduce_rule(('cos_sim', word1, 7, period_1))
        except Exception as e:
            logger.warning(f"Word '{word1}' not in vocabulary: {e}")
            yield ('error', f"The word '{word1}' is not available in the vocabulary. Please try a different word.")
            return

        try:
            ai_poet.introduce_rule(('cos_sim', word2, 10, period_2))
        except Exception as e:
            logger.warning(f"Word '{word2}' not in vocabulary: {e}")
            yield ('error', f"The word '{word2}' is not available in the vocabulary. Please try a different word.")
            return

        for verse_num, verse_text in ai_poet.generate_text_stream():
            yield (verse_num, verse_text)

    except Exception as e:
        logger.error(f"Poem generation failed: {e}")
        yield ('error', "An error occurred during poem generation. Please try again with different words.")

# Streamlit UI
st.set_page_config(
    layout="wide",
    page_title="WOPE - Wave Operated Poetry Engine",
    page_icon="🌊"
)

st.title("🌊 WOPE 🎭")
st.markdown("### *Wave Operated Poetry Engine*")

# Wave divider
st.markdown('<div class="wave-divider"></div>', unsafe_allow_html=True)

st.markdown("""
**WOPE** generates poetry by weaving together semantic concepts in wave-like patterns.
Choose two words and watch as they oscillate through your poem, creating unique thematic interplay.

💡 *Tip: Try contrasting concepts like "ocean" & "memory" or "chaos" & "silence"*

📖 [Read the research paper](https://computationalcreativity.net/iccc24/wp-content/uploads/2023/12/PerezBenavente_ECS_ICCC24.pdf)
""")

st.markdown('<div class="wave-divider"></div>', unsafe_allow_html=True)

# Two-column layout for semantic words
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌀 First Semantic Wave")
    word1 = st.text_input("Semantic Word 1:", placeholder="e.g., ocean, time, light...")
    num1 = st.slider("Wave Intensity:", 1, 10, 5, key="slider1",
                     help="Higher values make this concept more prominent")

with col2:
    st.markdown("### 🌊 Second Semantic Wave")
    word2 = st.text_input("Semantic Word 2:", placeholder="e.g., memory, silence, shadow...")
    num2 = st.slider("Wave Intensity:", 1, 10, 5, key="slider2",
                     help="Higher values make this concept more prominent")

st.markdown('<div class="wave-divider"></div>', unsafe_allow_html=True)

# Beginning text input
st.markdown("### ✍️ Opening Line")
beginning = st.text_input("Start your poem with:", placeholder="The ocean waves...")

st.markdown('<div class="wave-divider"></div>', unsafe_allow_html=True)

# Centered button layout
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_button = st.button("🌊 Generate Poem", use_container_width=True)

if generate_button:
    if word1 and word2 and beginning:
        # Clear previous poem
        st.session_state["poem"] = ""
        st.session_state["verses"] = []

        # Create placeholder for streaming output
        st.markdown('<div class="wave-divider"></div>', unsafe_allow_html=True)
        st.markdown("## ✨ Your Generated Poem")
        poem_container = st.empty()

        # Stream verses as they're generated
        full_poem = ""
        for verse_num, verse_text in generate_poem_stream(
            word1=word1,
            strength1=num1,
            word2=word2,
            strength2=num2,
            beginning=beginning
        ):
            if verse_num == 'error':
                st.error(f"⚠️ {verse_text}")
                break

            # Add verse to full poem
            if verse_num == 0:
                full_poem = verse_text
            else:
                full_poem += " " + verse_text

            # Update the display with current progress
            poem_container.markdown(
                f'<div class="poem-container">{full_poem}</div>',
                unsafe_allow_html=True
            )

        # Store final poem in session state
        st.session_state["poem"] = full_poem
    else:
        st.error("⚠️ Please enter both semantic words and an opening line!")

# Display reset button if poem exists
if "poem" in st.session_state and st.session_state["poem"]:
    # Reset button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🔄 Create Another", use_container_width=True):
            st.session_state.clear()
            st.rerun()

# Footer
st.markdown('<div class="wave-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #718096; padding: 2rem 0;'>
    <p>🌊 Powered by semantic wave modulation • Built with GPT-2 & PyTorch</p>
    <p style='font-size: 0.9rem;'>Research from ICCC 2024 Conference</p>
</div>
""", unsafe_allow_html=True)
