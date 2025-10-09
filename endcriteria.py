from transformers import NoRepeatNGramLogitsProcessor, MinNewTokensLengthLogitsProcessor, ForcedEOSTokenLogitsProcessor
from transformers import StoppingCriteria
from masterlogits import MasterLogits
import torch

class EndCriteria(StoppingCriteria):
    """
    Custom stopping criteria that terminates generation when a specific token is encountered.
    """

    def __init__(self, token):
        """
        Initializes the stopping criteria with a specific stop token.

        Args:
            token (int): Token ID that signals generation should stop (typically EOS token).
        """
        StoppingCriteria.__init__(self)
        self.stop_token = token

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """
        Checks if generation should stop based on the presence of the stop token.

        Args:
            input_ids (torch.LongTensor): Tokens generated so far for each beam.
            scores (torch.FloatTensor): Current logit scores (unused).

        Returns:
            bool: True if stop token is detected in any beam, False otherwise.
        """
        force_end = False

        for beam_index, beam_input_ids in enumerate(input_ids):
            if beam_input_ids[-1] == self.stop_token:
                force_end = True
                break
            else:
                force_end = False

        return force_end