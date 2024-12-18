from transformers import LogitsProcessor
from utils import ban_scores_syl, ban_scores, boost_scores

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
      self.syllable = False
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