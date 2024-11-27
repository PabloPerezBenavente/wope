#!/usr/bin/env python3
"""
pip install transformers
pip install torch
pip install numpy
pip install tensorflow
pip install ~/Downloads/tensorflow-2.4.1-py3-none-any.whl

pip install --use-pep517
arch -arm64 brew upgrade
arch -arm64 brew install zlib
brew install pyenv
"""
#haha


from poem_agent import PoemAgent



class Copoet(PoemAgent):
    def __init__(self, num_verses = 4, no_repeat = 1):
        super().__init__(num_verses=num_verses, no_repeat=no_repeat)

