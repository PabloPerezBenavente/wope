from transformers import NoRepeatNGramLogitsProcessor, MinNewTokensLengthLogitsProcessor, ForcedEOSTokenLogitsProcessor
from transformers import StoppingCriteria, LogitsProcessor
import torch
from utils import ban_scores, boost_scores

class MasterLogits(LogitsProcessor):
  """
   Master of all logits manipulations: boosts or bans tokens based on semantic similarity, timestep,
   and user-defined intensity. Think of it as the puppet master of word probabilities.
   """

  def __init__(self, LogitsProcessor_dict, vocab, tokenizer):
    """
    Initialize the MasterLogits processor with rules for boosting/banning tokens.
    LogitsProcessor_dict: contains all the information about how to manipulate scores.
    """

    self.vocab = vocab  # The full dictionary of words we’re working with
    self.tokenizer = tokenizer # Tokenizer to convert between words and tokens

    # Check if we're using cosine similarity for boosting
    if LogitsProcessor_dict['cos_sim']:
      self.similarity = True # Activate the semantic wizardry
      self.cos_sim_dict = LogitsProcessor_dict['cos_sim'] # Dictionary with keyword boosting rules
      self.keywords = self.cos_sim_dict.keys() # Keywords for boosting
      self.first_prediction = 0 # Track where the first boost happens
      self.current_verse = LogitsProcessor_dict['current_verse']  # Verse tracking (poetry-related context)
    else:
      self.similarity = False # No similarity boosting; keep it boring
      self.cos_sim_dict = False # No rules to apply

  def __call__(self, input_ids, scores):
    """
    Modify logits (scores) by boosting or banning tokens based on semantic similarity.
    Arguments:
        input_ids: Tokenized inputs for the current generation step.
        scores: The logits (scores) output by the model for all tokens.
    Returns:
        Modified scores with boosted/banned tokens.
    """

    tokens_to_ban = [] # Tokens that will be banned (blocked from generation)
    tokens_to_boost = [] # Tokens that will get a little extra "love" (boosted scores)

    for beam_index, beam_input_ids in enumerate(input_ids): # Loop over each beam of generation

      tokens_to_ban_in_one_beam = [] # Banned tokens for this beam
      semantic_tokens_to_boost_in_one_beam = [] # Boosted tokens for this beam

      if self.similarity: # If we're using semantic similarity rules

        # First prediction (special case: initialize boosting)
        if self.current_verse == 0 and self.first_prediction == 0:
          for keyword in self.keywords:
            # Boost tokens related to the keyword
            semantic_tokens_to_boost_in_one_beam.append(self.cos_sim_dict[keyword]['vocab'])
            # Store where the first boost occurred
            self.first_prediction = len(beam_input_ids)

        else:
          for keyword in self.keywords:
            # Boost tokens at specific timesteps based on keyword frequency
            if (len(beam_input_ids.tolist()) - self.first_prediction) % self.cos_sim_dict[keyword]['period'] == 0:
              semantic_tokens_to_boost_in_one_beam.append(self.cos_sim_dict[keyword]['vocab'])
            # Ban tokens otherwise to keep the output under control
            else:
              semantic_tokens_to_boost_in_one_beam.append([])
              tokens_to_ban_in_one_beam.extend(self.cos_sim_dict[keyword]['vocab'])

      # Collect all the tokens to ban/boost for this beam
      tokens_to_ban.append(tokens_to_ban_in_one_beam)
      tokens_to_boost.append(semantic_tokens_to_boost_in_one_beam)

    # Apply token bans to the scores (zero out banned tokens)
    scores = ban_scores(scores, tokens_to_ban)

    # Apply token boosts to the scores (give some tokens a popularity bump or a VIP pass)
    scores = boost_scores(scores, self.cos_sim_dict, tokens_to_boost, self.vocab)

    return scores

class EndCriteria(StoppingCriteria):
  """
  Custom stopping criteria for text generation.
  Stops generation when a specified token is generated.
  Think of it as the bouncer who kicks everyone out when the stop token shows up.
  """

  def __init__(self, token):
    """
    Initialize with the token that signals the end of generation.
    Arguments:
        token: The token (usually an integer) that stops the generation party.
    """
    StoppingCriteria.__init__(self)
    self.stop_token = token # The bouncer's "that's enough" signal

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
    """
    Check if the generation should stop.
    Arguments:
        input_ids: The tokens generated so far for each beam (list of lists).
        scores: The current scores (logits) for the tokens (not used here, just along for the ride).
    Returns:
        force_end: True if the stop token is detected in any beam, False otherwise.
    """
    force_end = False # Assume we're not done yet

    for beam_index, beam_input_ids in enumerate(input_ids): # Check each beam separately
      if beam_input_ids[-1] == self.stop_token: # If the last token matches the stop token
        force_end = True # Party's over, generation stops!
        break
      else:
        force_end = False # Keep generating

    return force_end # Signal whether to stop or keep going

def get_tool_tools(dict_of_tools, tokenizer, vocab):
  """
  Loads all the generation tools (logit processors and stopping criteria) into one neat package.
  Think of it as the toolbox that holds everything you need to manipulate, restrict, and stop generation.
  """
  tool_tools = {}

  # Number of verses we want (because poems need limits, unlike your imagination)
  tool_tools['num_verses'] = dict_of_tools['num_verses']

  # Create a slot for all the LogitsProcessors (fancy tools for controlling the model's word choices)
  tool_tools['LogitsProcessors'] = []

  # Create a slot for StoppingCriteria (the emergency brake for generation)
  tool_tools['StoppingCriteria'] = []

  # Set repetition restrictions (because no one likes repetitive poetry)
  if dict_of_tools['no_repeat']:
    # Prevent n-grams from repeating themselves like a broken record
    tool_tools['LogitsProcessors'] = [NoRepeatNGramLogitsProcessor(dict_of_tools['no_repeat'])]

  # Enforce fixed verse size using min/max token length (poetry with rules is still poetry)
  if dict_of_tools['verse_size']:
    # Special case for the first verse (get ready to start strong!)
    if dict_of_tools['current_verse'] == 0:
      tool_tools['LogitsProcessors'].append(
        MinNewTokensLengthLogitsProcessor(dict_of_tools['input_length'],
                                          dict_of_tools['verse_size'] + 1, # Add 1 for the <|endoftext|> token
                                          tokenizer.eos_token_id))
      tool_tools['LogitsProcessors'].append(
        ForcedEOSTokenLogitsProcessor(dict_of_tools['verse_size'] + 1,
                                      tokenizer.eos_token_id))
    # For subsequent verses (because consistency matters... sometimes)
    else:
      tool_tools['LogitsProcessors'].append(
        MinNewTokensLengthLogitsProcessor(dict_of_tools['input_length'],
                                          dict_of_tools['verse_size'] + dict_of_tools['input_length'] + 1,
                                          tokenizer.eos_token_id))
      tool_tools['LogitsProcessors'].append(
        ForcedEOSTokenLogitsProcessor(dict_of_tools['verse_size'] + dict_of_tools['input_length'] + 1,
                                      tokenizer.eos_token_id))
  else:
    # If no verse size is specified and it's not the first verse, set a minimum token length
    if dict_of_tools['current_verse'] != 0:
      tool_tools['LogitsProcessors'].append(
        MinNewTokensLengthLogitsProcessor(dict_of_tools['input_length'],
                                          2, # Minimum 2 tokens (because a verse with 1 word isn't impressive)
                                          tokenizer.eos_token_id))

  # Apply semantic restrictions (aka the "poetry police")
  if dict_of_tools['cos_sim']:
    tool_tools['LogitsProcessors'].append(
      MasterLogits(dict_of_tools, vocab, tokenizer)) # The brains behind semantic similarity magic

  # Set stopping criteria (stop when <|endoftext|> shows up — because all things good come to an end)
  tool_tools['StoppingCriteria'].append(
    EndCriteria(tokenizer.eos_token_id))

  return tool_tools # Return the fully loaded toolbox, ready for action