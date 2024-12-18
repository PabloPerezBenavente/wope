import pronouncing
import torch
from transformers import BeamSearchScorer, \
    LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria

from toolprocessors import get_tool_tools
from utils import get_syllables_in_verse

def generator(prompt, active_tools, tokenizer, vocab, model):

    # how many beams to track during the Viterbi algorithm
    num_beams = 10
    # how many beams to return after the algorithm
    num_return_beams = 5

    input = prompt
    poem = {}
    for num_verse in range(active_tools['num_verses']):

        # save current verse
        active_tools['current_verse'] = num_verse

        # transform the words so far into token ids
        prompt_tokenized = tokenizer(input, return_tensors='pt' )['input_ids']

        # get instances of logits processor and stopping criteria to use during beam search
        tool_tools = get_tool_tools(active_tools, tokenizer, vocab)
        logits_processor = LogitsProcessorList(tool_tools['LogitsProcessors'])
        stopping_criteria = StoppingCriteriaList(tool_tools['StoppingCriteria'])

        # instantiating a BeamSearchScorer
        beam_scorer = BeamSearchScorer(
            batch_size = prompt_tokenized.shape[0],
            num_beams = num_beams,
            num_beam_hyps_to_keep = num_return_beams,
            device=model.device
        )

        # running beam search using our custom LogitsProcessor
        generated = model.beam_search(
            torch.cat([prompt_tokenized] * num_beams),
            beam_scorer,
            logits_processor=logits_processor,
            pad_token_id=0,
            stopping_criteria=stopping_criteria
        )

        # save predictions
        predictions = generated

        # get list of tokens until <|endoftext|>
        clean_predictions = [beam.tolist()[0:beam.tolist().index(50256)] for beam in predictions]

        # get beam with highest score
        verse_tokens = clean_predictions[0]

        # remove <|endoftext|> so that next verse continues this one
        try:
            input = tokenizer.decode(verse_tokens[0:verse_tokens.index(50256)])
        except ValueError as e:
            input = tokenizer.decode(verse_tokens)

        # set length of input for next verse
        active_tools['input_length'] = len(verse_tokens)
        if num_verse == 0:
            verse = tokenizer.decode(verse_tokens)
        else:
            verse = tokenizer.decode(verse_tokens[len(prompt_tokenized[0]):])

        poem[f'VERSE {num_verse}'] = verse
        poem[f'VERSE {num_verse} tokens'] = verse_tokens

    return poem