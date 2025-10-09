import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import get_semantic_items
from poem_generator import generator
from constants import DEFAULT_NUM_VERSES, DEFAULT_VERSE_SIZE, DEFAULT_NO_REPEAT_NGRAM, DEFAULT_INTENSITY, DEFAULT_PERIOD

class PoemAgent():
  def __init__(self, num_verses=DEFAULT_NUM_VERSES, no_repeat=DEFAULT_NO_REPEAT_NGRAM, verse_size=DEFAULT_VERSE_SIZE):

    # util functions
    self.generator = generator

    # define model, vocabulary and tokenizer
    self.model = GPT2LMHeadModel.from_pretrained("gpt2")
    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    self.vocab = self.tokenizer.encoder

    # create workspace
    self.active_tools = {}
    self.tool_tools = []

    # library of possible tools
    self.tool_lib = ['cos_sim', 'verse_size', 'num_verses', 'no_repeat', 'blacklist']

    # initialise workspace
    self.active_tools = {tool : False for tool in self.tool_lib}

    # define default configurations
    self.active_tools['num_verses'] = num_verses
    self.active_tools['no_repeat'] = no_repeat
    self.active_tools['verse_size'] = verse_size

    # load valid vocabulary
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_dir = os.path.join(current_dir, 'data', 'vocab')

    with open(os.path.join(vocab_dir, 'GPT2_tokens_in_cmu'), 'r') as f:
      self.word_tokens_in_cmu = f.read().split('\n')[:-1]
    with open(os.path.join(vocab_dir, 'GPT2_tokens_not_in_cmu'), 'r') as f:
      self.word_tokens_not_in_cmu = f.read().split('\n')[:-1]
    self.active_tools.update({'word_tokens_not_in_cmu' : self.word_tokens_not_in_cmu, 'word_tokens_in_cmu' : self.word_tokens_in_cmu})

  def check_workspace(self):
    output = {"message": self.active_tools}
    return output

  def introduce_rule(self, rule):

    # check the rule is valid
    if rule[0] not in self.tool_lib:
      e = "that tool doesn't exist"
      raise Exception(e)

    if rule[0] == 'cos_sim':
      # check that the input word is contained in vocabulary
      for token in list(self.vocab.keys()):
        if rule[1] == token.strip(' Ġ'):
          if not self.active_tools['cos_sim']:
            self.active_tools['cos_sim'] = {}

          # if word was already in memory, only update wave parameters
          if token in self.active_tools['cos_sim'].keys():
            if len(rule) == 4:
              self.active_tools['cos_sim'][token].update({'intensity': int(rule[2]), 'period': int(rule[3])})

          # if word was not in memory, create a slot, compute vocab of similar tokens and save wave parameters
          else:
            self.active_tools['cos_sim'].update({token: {'vocab': get_semantic_items(token, self.vocab, self.model)}})
            if len(rule) == 4:
              self.active_tools['cos_sim'][token].update({'intensity': int(rule[2]), 'period': int(rule[3])})
            if len(rule) == 2:
              self.active_tools['cos_sim'][token].update({'intensity': DEFAULT_INTENSITY, 'period': DEFAULT_PERIOD})

          return True

      # raise error if word isn't contained in vocab
      e = "the specified word isn't contained within GPT2's vocab"
      raise Exception(e)

    if rule[0] == 'verse_size':
      self.active_tools['verse_size'] = rule[1]
      return True

    if rule[0] == 'no_repeat':
      self.active_tools[rule[0]] = rule[1]
      return True


    if rule[0] == 'num_verses':
      self.active_tools['num_verses'] = rule[1]
      return True

  def eliminate_rule(self, rule):

    if self.active_tools[rule] and rule != 'cos_sim':
      self.active_tools[rule]['active'] = False
    elif self.active_tools[rule] and rule == 'cos_sim':
      self.active_tools[rule] = False
    else:
      e = "this tool wasn't active"
      raise Exception(e)

  def create_input(self, input):

    # save input and number of words
    self.input = input
    self.input_length = len(input.split(' '))
    self.active_tools['input_length'] = self.input_length

  def generate_text(self):

    if not self.input:
      return False

    poem = self.generator(self.input, self.active_tools, self.tokenizer, self.vocab, self.model)
    poem = ' '.join([poem[verse] for verse in poem.keys() if 'tokens' not in verse])

    self.tool_tools = {}

    return poem

  def generate_text_stream(self):
    """
    Generates poem verse by verse, yielding each verse as it's created.

    Yields:
        tuple: (verse_num, verse_text) for each generated verse
    """
    if not self.input:
      return

    for verse_num, verse_text, verse_tokens in self.generator(
        self.input, self.active_tools, self.tokenizer, self.vocab, self.model, stream=True
    ):
      yield (verse_num, verse_text)

    self.tool_tools = {}

