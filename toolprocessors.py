from transformers import NoRepeatNGramLogitsProcessor, MinNewTokensLengthLogitsProcessor, ForcedEOSTokenLogitsProcessor
from stopping_criteria_classes import EndCriteria
from constrained_generation import MasterLogits

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