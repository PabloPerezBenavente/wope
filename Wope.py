#!/usr/bin/env python3
from poem_agent import PoemAgent

class Copoet(PoemAgent):
    def __init__(self, num_verses = 4, no_repeat = 1):
        super().__init__(num_verses=num_verses, no_repeat=no_repeat)

