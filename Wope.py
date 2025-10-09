#!/usr/bin/env python3
from poem_agent import PoemAgent
from constants import DEFAULT_NUM_VERSES, DEFAULT_NO_REPEAT_NGRAM

class Copoet(PoemAgent):
    def __init__(self, num_verses=DEFAULT_NUM_VERSES, no_repeat=DEFAULT_NO_REPEAT_NGRAM):
        super().__init__(num_verses=num_verses, no_repeat=no_repeat)

