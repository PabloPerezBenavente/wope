from transformers import StoppingCriteria, LogitsProcessor
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