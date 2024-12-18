from transformers import NoRepeatNGramLogitsProcessor, MinNewTokensLengthLogitsProcessor, ForcedEOSTokenLogitsProcessor
git from transformers import StoppingCriteria
import torch
from transformers import LogitsProcessor
from utils import ban_scores, boost_scores

class MasterLogits(LogitsProcessor):
  """
  LogitsProcessor instance that performs all logits manipulations and processing before emitting output
  args: LogitsProcessor_dict: dictionary with the information to manipulate the logits.
  """
  def __init__(self, LogitsProcessor_dict, vocab, tokenizer):

    self.vocab = vocab
    self.tokenizer = tokenizer

    # semantic config
    if LogitsProcessor_dict['cos_sim']:
      self.similarity = True
      self.cos_sim_dict = LogitsProcessor_dict['cos_sim']
      self.keywords = self.cos_sim_dict.keys()
      self.first_prediction = 0
      self.current_verse = LogitsProcessor_dict['current_verse']
    else:
      self.similarity=False
      self.cos_sim_dict = False

  def __call__(self, input_ids, scores):

    tokens_to_ban = []
    tokens_to_boost = []

    for beam_index, beam_input_ids in enumerate(input_ids):

      tokens_to_ban_in_one_beam = []
      semantic_tokens_to_boost_in_one_beam = []

      if self.similarity:

        # if this is the first prediction
        if self.current_verse == 0 and self.first_prediction == 0:
          for keyword in self.keywords:
            # we boost tokens
            semantic_tokens_to_boost_in_one_beam.append(self.cos_sim_dict[keyword]['vocab'])
            # we store location of first boost
            self.first_prediction = len(beam_input_ids)

        else:
          for keyword in self.keywords:
            # if current prediction is at time step that is multiple of keyword frequency, boost tokens related to keyword
            if (len(beam_input_ids.tolist()) - self.first_prediction) % self.cos_sim_dict[keyword]['period'] == 0:
              semantic_tokens_to_boost_in_one_beam.append(self.cos_sim_dict[keyword]['vocab'])
            # if current prediction doesn't match with keyword frequency, ban tokens related to keyword
            else:
              semantic_tokens_to_boost_in_one_beam.append([])
              tokens_to_ban_in_one_beam.extend(self.cos_sim_dict[keyword]['vocab'])

      tokens_to_ban.append(tokens_to_ban_in_one_beam)
      tokens_to_boost.append(semantic_tokens_to_boost_in_one_beam)

    scores = ban_scores(scores, tokens_to_ban)
    scores = boost_scores(scores, self.cos_sim_dict, tokens_to_boost, self.vocab)

    return scores

class EndCriteria(StoppingCriteria):
  def __init__(self, token):
    StoppingCriteria.__init__(self)
    self.stop_token = token

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):

    force_end = False

    for beam_index, beam_input_ids in enumerate(input_ids):
      if beam_input_ids[-1] == self.stop_token:
        force_end = True
      else:
        force_end = False

    return force_end

def get_tool_tools(dict_of_tools, tokenizer, vocab):
  tool_tools = {}

  # set number of lines
  tool_tools['num_verses'] = dict_of_tools['num_verses']

  # create slot for Logit Processors
  tool_tools['LogitsProcessors'] = []

  # create a slot for Stopping Criteria
  tool_tools['StoppingCriteria'] = []

  # set repetition restrictions
  if dict_of_tools['no_repeat']:
    tool_tools['LogitsProcessors'] = [NoRepeatNGramLogitsProcessor(dict_of_tools['no_repeat'])]

  # fixed size through max and min. We add 1 to consider <|endoftext token|>
  if dict_of_tools['verse_size']:

    if dict_of_tools['current_verse'] == 0:
      tool_tools['LogitsProcessors'].append(MinNewTokensLengthLogitsProcessor(dict_of_tools['input_length'], dict_of_tools['verse_size'] + 1, tokenizer.eos_token_id))
      tool_tools['LogitsProcessors'].append(ForcedEOSTokenLogitsProcessor(dict_of_tools['verse_size'] + 1, tokenizer.eos_token_id))
    else:
      tool_tools['LogitsProcessors'].append(MinNewTokensLengthLogitsProcessor(dict_of_tools['input_length'], dict_of_tools['verse_size'] + dict_of_tools['input_length'] + 1, tokenizer.eos_token_id))
      tool_tools['LogitsProcessors'].append(ForcedEOSTokenLogitsProcessor(dict_of_tools['verse_size'] + dict_of_tools['input_length'] + 1, tokenizer.eos_token_id))
  else:
    if dict_of_tools['current_verse'] != 0:
      tool_tools['LogitsProcessors'].append(MinNewTokensLengthLogitsProcessor(dict_of_tools['input_length'], 2, tokenizer.eos_token_id))

  # apply semantic, syllabic and rhyming restrictins
  if dict_of_tools['cos_sim']:
    tool_tools['LogitsProcessors'].append(MasterLogits(dict_of_tools, vocab, tokenizer))

  # set end criteria
  tool_tools['StoppingCriteria'].append(EndCriteria(tokenizer.eos_token_id))


  return tool_tools