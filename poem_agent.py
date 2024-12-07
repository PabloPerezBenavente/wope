import pronouncing
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from logits_processor import SemLogits
from utils import get_word_tokens_not_in_cmu, rhymes_to_numeric_tokens, get_syl_items, get_semantic_items, get_rhyme_and_syl_data
from poem_generator import generator
import json



class PoemAgent():
  def __init__(self, num_verses = 4, no_repeat=1, verse_size = 20):

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
    self.tool_lib = ['cos_sim', 'verse_size', 'num_verses', ' no_repeat', 'blacklist', 'num_syl', 'rhyme', 'scheme']

    # initialise workspace
    self.active_tools = {tool : False for tool in self.tool_lib}

    # define default configurations
    self.active_tools['num_verses'] = num_verses
    self.active_tools['no_repeat'] = no_repeat
    self.active_tools['verse_size'] = verse_size

    print('default configuration: \n number of lines = 4 \n no other restrictions')

    # load valid vocabulary
    with open('data/vocab/GPT2_tokens_in_cmu', 'r') as f:
      self.word_tokens_in_cmu = f.read().split('\n')[:-1]
    with open('data/vocab/GPT2_tokens_not_in_cmu', 'r') as f:
      self.word_tokens_not_in_cmu = f.read().split('\n')[:-1]
    self.active_tools.update({'word_tokens_not_in_cmu' : self.word_tokens_not_in_cmu, 'word_tokens_in_cmu' : self.word_tokens_in_cmu})

    # create slot for rhyme
    self.active_tools['rhyme'] = {'active' : False}

    # dictionary of words sorted by rhyming part, dictionary of tokens that dont rhyme sorted by syllable
    # and dictionary of tokens with polisyllable rhyme, also sorted by syllable
    with open('data/rhyme/tokens_that_rhyme', 'r') as f:
      self.rhyme_dict_to_numeric_tokens = json.loads(f.read())

    # create slot for syllables
    self.active_tools['num_syl'] = {'active' : False}

    # dictionary of tokens with their syllable count, and dictionary of tokens sorted by syllable
    with open('data/meter/tokens_to_number_of_syllables', 'r') as f:
      self.numeric_tokens_to_syl = json.loads(f.read())
    with open('data/meter/number_of_syllables_to_tokens', 'r') as f:
      self.syl_to_numeric_tokens = json.loads(f.read()) # WARNING: this reads token ids as strings rather than ints, which might cause problems somewhere down the pipeline
    self.active_tools['num_syl'].update({'numeric_tokens_to_syl' : self.numeric_tokens_to_syl, 'syl_to_numeric_tokens' : self.syl_to_numeric_tokens})

    # get combined data of rhymes and syllables
    with open('data/rhyme/tokens_with_rhyme_and_syllable_count', 'r') as f:
      self.rhyme_and_syl_dict = json.loads(f.read())
    with open('data/rhyme/no_rhyme_and_syl_data', 'r') as f:
      self.no_rhyme_and_syl_dict = json.loads(f.read())
    self.active_tools['rhyme'].update({'rhyme_and_syl_dict': self.rhyme_and_syl_dict, 'no_rhyme_syl_dict': self.no_rhyme_and_syl_dict})

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
              self.active_tools['cos_sim'][token].update({'intensity': int(6), 'period': int(3)})

          return True

      # raise error if word isn't contained in vocab
      e = "the specified word isn't contained within GPT2's vocab"
      raise Exception(e)

    if rule[0] == 'verse_size':
      self.active_tools['verse_size'] = rule[1]

      return True

    if rule[0] == 'rhyme':

      # if rhyme is set in 'hard mode' group verses by rhyme type
      if type(rule[1]) == list:

        # WARNING: 'active' key should be unnecessary
        self.active_tools['rhyme'].update({'active' : True, 'type' : 'hard', 'list_scheme' : rule[1], 'rhyme_scheme' : {}})
        for n, rhyme_type in enumerate(rule[1]):
          try:
            self.active_tools['rhyme']['rhyme_scheme'][rhyme_type]['verses'].append(n)
          except KeyError:
            self.active_tools['rhyme']['rhyme_scheme'][rhyme_type] = {'verses' : [n], 'rhyming_words' : [], 'rhyming_part' : []}

      # if rhyme is set in 'soft mode' (to be implemented)
      else:
        self.active_tools['rhyme'].update({'type' : 'soft', 'rhyme_scheme' : {}})

      return True

    if rule[0] == 'num_syl':

      # discard max number of tokens
      self.active_tools['verse_size'] = False

      # create syllable slot
      # WARNING: 'active' key should be unnecessary
      self.active_tools['num_syl'].update({'active' : True, 'number' : rule[1]})

      return True

    if rule[0] == 'no_repeat':

      self.active_tools[rule[0]] = rule[1]

      return True

    # 'scheme' and 'num_syl' rules are introduced differently but point under the hood to the same process but
    # with a different datatype. How pythonic is this?
    if rule[0] == 'scheme':

      # discard max number of tokens
      self.active_tools['verse_size'] = False
      # save number of verses
      self.active_tools['num_verses'] = len(rule[1])
      # create syllable slot
      self.active_tools['num_syl'].update({'active' : True, 'scheme' : rule[1]})

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
      print("this rule wasn't active")

  def create_input(self, input):
    self.input = input
    self.input_length = len(input.split(' '))
    self.active_tools['input_length'] = self.input_length

    if self.active_tools['num_syl']:
      self.input_syl = 0
      for word in input.split(' '):
        try:
          self.input_syl += pronouncing.syllable_count(pronouncing.phones_for_word(word)[0])
        except:
          print('probably, generated word was not in cmu')
      self.active_tools['num_syl']['input_syl'] = self.input_syl

  def generate_text(self):

    if not self.input:
      print('you forgot to introduce the input')
      return False

    else:

      print('your input is: ', self.input)
      print('generating with: ', list(self.active_tools.keys()))
      print('cos sim value: ', self.active_tools['cos_sim'])
      print('verse size value: ', self.active_tools['verse_size'])
      print('num verses: ', self.active_tools['num_verses'])
      #print('num syl: ', self.active_tools['num_syl'])

      poem = self.generator(self.input, self.active_tools, self.tokenizer, self.vocab, self.model)

      print(poem)

      self.tool_tools = {}

      #clean_poem = []
      #if type(poem) == dict:
        #for n, verse in enumerate(poem.keys()):
          #if len(verse) < 8:
            #clean_poem.append(poem[verse])

      #return clean_poem
