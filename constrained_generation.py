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

    else:
      self.similarity=False
      self.cos_sim_dict = False


    # if using syllable or rhyme restrictions,
    # get active vocabulary (in CMU)
    if LogitsProcessor_dict['num_syl']['active'] or LogitsProcessor_dict['rhyme']['active']:
      self.word_tokens_not_in_cmu = LogitsProcessor_dict['word_tokens_not_in_cmu']
      self.numeric_tokens_not_in_cmu = [self.vocab[word_token] for word_token in self.word_tokens_not_in_cmu]
      for i in range(32, 57):
        self.numeric_tokens_not_in_cmu.append(i)
      self.word_tokens_in_cmu = LogitsProcessor_dict['word_tokens_in_cmu']

    # syllabic config
    if LogitsProcessor_dict['num_syl']['active']:
      self.current_verse = LogitsProcessor_dict['current_verse']
      self.syllable = True
      self.syl_data = LogitsProcessor_dict['num_syl']
      if self.current_verse != 0:
        try:
          self.length_of_verse_in_syllables = self.syl_data['scheme'][self.current_verse] + self.syl_data['input_syl']
        except KeyError:
          self.length_of_verse_in_syllables = self.syl_data['number']*(self.current_verse + 1)
      else:
        try:
          self.length_of_verse_in_syllables = self.syl_data['scheme'][self.current_verse]
        except KeyError:
          self.length_of_verse_in_syllables = self.syl_data['number']*(self.current_verse + 1)
      self.numeric_tokens_to_syl = self.syl_data['numeric_tokens_to_syl']
      self.syl_to_numeric_tokens = self.syl_data['syl_to_numeric_tokens']
      self.syl_in_verse = 0

    else:
      self.syllable = False
      self.out_length = [LogitsProcessor_dict['verse_size'] + LogitsProcessor_dict['input_length']][0]


    # rhyme_config
    if LogitsProcessor_dict['rhyme']['active']:
      self.rhyme = True
      self.rhyme_data = LogitsProcessor_dict['rhyme']
      self.rhyme_type = self.rhyme_data['type']
    else:
      self.rhyme = False




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
              #print(f"boosting at timestep {len(beam_input_ids.tolist())}")

            # if current prediction doesnt match with keyword frequency, ban tokens related to keyword
            else:
              semantic_tokens_to_boost_in_one_beam.append([])
              tokens_to_ban_in_one_beam.extend(self.cos_sim_dict[keyword]['vocab'])
              #print(f"banning at timestep {len(beam_input_ids.tolist())}")

      if self.syllable or self.rhyme:
        tokens_to_ban_in_one_beam.extend(self.numeric_tokens_not_in_cmu[:-1])
        #if 45885 in tokens_to_ban_in_one_beam:
          #print('crotch was banned because its not in cmu')

      tokens_to_ban.append(tokens_to_ban_in_one_beam)
      tokens_to_boost.append(semantic_tokens_to_boost_in_one_beam)

    #print(f"size of scors before semantic is {scores.size()}")
    scores = ban_scores(scores, tokens_to_ban)
    scores = boost_scores(scores, self.cos_sim_dict, tokens_to_boost, self.vocab)

    #print(f"size of scores after semantic is {scores.size()}")

    tokens_to_ban = []
    tokens_to_boost = []

    for beam_index, beam_input_ids in enumerate(input_ids):

      tokens_to_ban_in_one_beam = []
      phonetic_tokens_to_boost_in_one_beam = []

      if self.syllable:

        if self.rhyme and self.rhyme_type == 'hard':
          tokens_to_ban_rhymesyl, tokens_to_boost_rhymesyl = ban_scores_syl(self.tokenizer, beam_input_ids, self.numeric_tokens_to_syl, self.syl_to_numeric_tokens,
                                         self.length_of_verse_in_syllables, rhyme = self.rhyme_data, rhyme_type = 'hard',
                                         current_verse = self.current_verse)

          #print(f'and the type of tokens at beam {beam_index} when rhyme is: ', type(tokens_to_ban_rhymesyl))

        elif not self.rhyme:
          tokens_to_ban_rhymesyl, tokens_to_boost_rhymesyl = ban_scores_syl(self.tokenizer, beam_input_ids, self.numeric_tokens_to_syl, self.syl_to_numeric_tokens,
                                         self.length_of_verse_in_syllables, rhyme = False, rhyme_type = False,
                                         current_verse = self.current_verse)

          #print(f'and the type of tokens at beam {beam_index} when no rhyme is:', type(tokens_to_ban_rhymesyl))


        elif self.rhyme and self.rhyme_type == 'soft':
          tokens_to_ban_rhymesyl = ban_scores_syl(self.tokenizer, beam_input_ids, self.numeric_tokens_to_syl, self.syl_to_numeric_tokens,
                                         self.length_of_verse_in_syllables, rhyme = self.rhyme_data, rhyme_type = 'soft',
                                         current_verse = self.current_verse)

        elif self.rhyme and not self.syllable:
          if self.rhyme_type == 'hard':

            # if this will be the last token and it is not the first verse
             if len(beam_input_ids) == self.out_length - 1 and self.current_verse != 0:
                ###
                ###
                ### find a different way of implementing this
                ###
                ###
              tokens_to_ban_rhymesyl = self.numeric_tokens_that_dont_rhyme
          elif self.rhyme_type == 'soft':
            pass
            ###
            ###
            ###   IMPLEMENT SOFT RHYME
            ###
            ###
            ###
            ############################
        else:
          tokens_to_ban_rhymesyl = []

        #print('the type of tokens to ban: ', type(tokens_to_ban_rhymesyl))
        #print('the type of tokens to boost: ', type(tokens_to_boost_rhymesyl))
        tokens_to_ban_in_one_beam = tokens_to_ban_rhymesyl
        phonetic_tokens_to_boost_in_one_beam = tokens_to_boost_rhymesyl

        flag_tokens = [1262, 3500]

        for token in flag_tokens:
          if token in tokens_to_ban_in_one_beam:
            pass
            #print(f"{token} has been banned")

      tokens_to_boost.append(phonetic_tokens_to_boost_in_one_beam)
      tokens_to_ban.append(tokens_to_ban_in_one_beam)

    #print('size of scores before sylable: ', scores.size())

    scores = ban_scores(scores, tokens_to_ban)

    #if self.similarity:

    scores = boost_scores(scores, False, tokens_to_boost, self.vocab)

    #print('size of scores after sylable: ', scores.size())

    return scores