import streamlit as st
from Wope import Copoet
ai_poet = Copoet()

def generate_poem(word1=None,
                  strength1=None,
                  word2=None,
                  strength2=None,
                  beginning=None):
    """Generates a simple poem using the two input words."""
    ai_poet.create_input(beginning)
    period_1 = 11 - strength1
    period_2 = 11 - strength2
    ai_poet.introduce_rule(('num_verses', 4))
    try:
        ai_poet.introduce_rule(('cos_sim', word1, 7, period_1))
    except Exception:
        return f'Oops! {word1} is probably too complex for me. Try again with something simpler :)'
    try:
        ai_poet.introduce_rule(('cos_sim', word2, 10, period_2))
    except Exception:
        return f'Oops! {word2} is probably too complex for me. Try again with something simpler :)'
    poem = ai_poet.generate_text()

    return poem

# Streamlit UI
st.set_page_config(
    layout="wide",
)
st.title("WOPE 🎭")
st.write("Wope generates text by combining two different realms, or domains, or words. This sometimes leads to freaky combinations and concepts that can serve as inspiration for poetic compositions! Just type in two words -unfortunately, complex words might not be available. If you dont like the resulting text, you can try adjusting the words' prominence with the sliders. [Here you can read more about this project](https://computationalcreativity.net/iccc24/wp-content/uploads/2023/12/PerezBenavente_ECS_ICCC24.pdf)")

# Input fields
word1 = st.text_input("Enter the first word:")
num1 = st.slider("Select a number for the first word:", 1, 10, 5)
word2 = st.text_input("Enter the second word:")
num2 = st.slider("Select a number for the second word:", 1, 10, 5)
beginning = st.text_input("Enter the beginning")

# Generate button
if st.button("Generate Poem"):
    if word1 and word2 and beginning:
        poem = generate_poem(word1=word1,
                             strength1=num1,
                             word2=word2,
                             strength2=num2,
                             beginning=beginning)
        st.session_state["poem"] = poem  # Store in session state
    else:
        st.error("Please enter both words and a beginning!")

# Display poem if generated
if "poem" in st.session_state:
    st.subheader("Your Generated Poem:")
    st.write(st.session_state["poem"])

# Reset button
if st.button("Reset"):
    st.session_state.clear()
    st.experimental_rerun()
