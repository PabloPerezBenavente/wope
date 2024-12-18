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