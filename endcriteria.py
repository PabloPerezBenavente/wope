from transformers import NoRepeatNGramLogitsProcessor, MinNewTokensLengthLogitsProcessor, ForcedEOSTokenLogitsProcessor
from transformers import StoppingCriteria
from masterlogits import MasterLogits
import torch

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