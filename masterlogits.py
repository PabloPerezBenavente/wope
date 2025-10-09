from transformers import StoppingCriteria, LogitsProcessor
from utils import ban_scores, boost_scores

class MasterLogits(LogitsProcessor):
    """
    Custom logits processor implementing semantic similarity wave modulation.

    Boosts or bans tokens based on semantic similarity to keywords at periodic
    intervals, creating wave-like patterns in the generated text.
    """

    def __init__(self, LogitsProcessor_dict, vocab, tokenizer):
        """
        Initializes the processor with semantic similarity configuration.

        Args:
            LogitsProcessor_dict (dict): Configuration containing cos_sim rules and verse tracking.
            vocab (dict): Complete vocabulary dictionary.
            tokenizer: Tokenizer for text encoding/decoding.
        """
        self.vocab = vocab
        self.tokenizer = tokenizer

        # Configure semantic similarity wave modulation
        if LogitsProcessor_dict['cos_sim']:
            self.similarity = True
            self.cos_sim_dict = LogitsProcessor_dict['cos_sim']
            self.keywords = self.cos_sim_dict.keys()
            self.first_prediction = 0  # Track position of first token prediction
            self.current_verse = LogitsProcessor_dict['current_verse']
        else:
            self.similarity = False
            self.cos_sim_dict = False

    def __call__(self, input_ids, scores):
        """
        Applies semantic similarity wave modulation to logit scores.

        Tokens are boosted at periodic intervals based on semantic similarity to keywords,
        and banned at other timesteps to create wave-like patterns.

        Args:
            input_ids (torch.Tensor): Tokenized inputs for current generation step.
            scores (torch.Tensor): Logit scores output by the model.

        Returns:
            torch.Tensor: Modified scores with boosted/banned tokens.
        """
        tokens_to_ban = []
        tokens_to_boost = []

        for beam_index, beam_input_ids in enumerate(input_ids):
            tokens_to_ban_in_one_beam = []
            semantic_tokens_to_boost_in_one_beam = []

            if self.similarity:
                # First prediction: initialize boosting for all keywords
                if self.current_verse == 0 and self.first_prediction == 0:
                    for keyword in self.keywords:
                        semantic_tokens_to_boost_in_one_beam.append(self.cos_sim_dict[keyword]['vocab'])
                        self.first_prediction = len(beam_input_ids)

                # Subsequent predictions: apply periodic boosting/banning
                else:
                    for keyword in self.keywords:
                        # Calculate position relative to first prediction
                        timestep = len(beam_input_ids.tolist()) - self.first_prediction

                        # Boost tokens at periodic intervals
                        if timestep % self.cos_sim_dict[keyword]['period'] == 0:
                            semantic_tokens_to_boost_in_one_beam.append(self.cos_sim_dict[keyword]['vocab'])
                        # Ban tokens at other timesteps
                        else:
                            semantic_tokens_to_boost_in_one_beam.append([])
                            tokens_to_ban_in_one_beam.extend(self.cos_sim_dict[keyword]['vocab'])

            # Collect tokens to ban/boost for this beam
            tokens_to_ban.append(tokens_to_ban_in_one_beam)
            tokens_to_boost.append(semantic_tokens_to_boost_in_one_beam)

        # Apply banning and boosting operations
        scores = ban_scores(scores, tokens_to_ban)
        scores = boost_scores(scores, self.cos_sim_dict, tokens_to_boost, self.vocab)

        return scores