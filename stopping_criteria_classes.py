from transformers import StoppingCriteria
import torch

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


class SylStoppingCriteria(StoppingCriteria):
  def __init__(self, active_tools, tokenizer):
    StoppingCriteria.__init__(self)

    self.tokenizer = tokenizer
    self.current_verse = active_tools['current_verse']
    if self.current_verse == 0:
      try:
        self.syl_num = active_tools['num_syl']['scheme'][self.current_verse]# + active_tools['num_syl']['input_syl']
      except KeyError:
        self.syl_num = active_tools['num_syl']['number']
    else:
      try:
        self.syl_num = active_tools['num_syl']['input_syl'] + active_tools['num_syl']['scheme'][self.current_verse]
      except KeyError:
        self.syl_num = active_tools['num_syl']['number']*(self.current_verse + 1)
    #print('next verse will be ', self.syl_num, ' syllables long')

    self.word_tokens_in_cmu = active_tools['word_tokens_in_cmu']
    self.word_tokens_not_in_cmu = active_tools['word_tokens_not_in_cmu']
    self.syl_to_numeric_tokens = active_tools['num_syl']['syl_to_numeric_tokens']
    self.numeric_tokens_to_syl = active_tools['num_syl']['numeric_tokens_to_syl']

  def __call__(self, input_ids, scores):
    #print('the number of syllables has to be', self.syl_num)

    stop = False
    beam_one = False
    beam_two = False

    stops = [False for beam in input_ids]

    for beam_index, beam_input_ids in enumerate(input_ids):

      syl_count = 0

      for token in beam_input_ids.tolist():
        if self.tokenizer.decode(token)[0] != "'" and self.tokenizer.decode(token) != 's':
          try:
            syl_count += self.numeric_tokens_to_syl[token]
            if syl_count >= self.syl_num:
              #print(f"stopping with generated: ", tokenizer.decode(beam_input_ids))
              stops[beam_index] = True
          except KeyError:
            for n, key in enumerate(self.numeric_tokens_to_syl.keys()):
              if n == 0:
                pass
            pass


    if stops.count(True) > len(stops) // 2:
      stop = True



    return stop
