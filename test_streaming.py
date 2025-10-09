#!/usr/bin/env python3
"""Test script to verify verse-by-verse streaming functionality."""

from Wope import Copoet

def test_streaming():
    """Test the new streaming generation method."""
    # Initialize the poet
    ai_poet = Copoet()

    # Configure the poem
    ai_poet.create_input("The ocean waves")
    ai_poet.introduce_rule(('num_verses', 4))
    ai_poet.introduce_rule(('cos_sim', 'ocean', 7, 5))
    ai_poet.introduce_rule(('cos_sim', 'memory', 10, 6))

    # Test streaming generation
    verses = []
    for verse_num, verse_text in ai_poet.generate_text_stream():
        verses.append(verse_text)

    return ' '.join(verses)

if __name__ == "__main__":
    test_streaming()
