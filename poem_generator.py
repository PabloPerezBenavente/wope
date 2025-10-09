from transformers import LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria
from transformers import NoRepeatNGramLogitsProcessor, MinNewTokensLengthLogitsProcessor, ForcedEOSTokenLogitsProcessor
from masterlogits import MasterLogits
from endcriteria import EndCriteria
from constants import NUM_BEAMS, NUM_RETURN_BEAMS, EOS_TOKEN_ID, PAD_TOKEN_ID


def get_tool_tools(dict_of_tools, tokenizer, vocab):
    """
    Constructs logits processors and stopping criteria based on generation configuration.

    Args:
        dict_of_tools (dict): Configuration dictionary containing generation parameters.
        tokenizer: Tokenizer with EOS token information.
        vocab (dict): Complete vocabulary dictionary.

    Returns:
        dict: Dictionary containing configured LogitsProcessors and StoppingCriteria lists.
    """
    tool_tools = {}

    # Store number of verses to generate
    tool_tools['num_verses'] = dict_of_tools['num_verses']

    # Initialize lists for logits processors and stopping criteria
    tool_tools['LogitsProcessors'] = []
    tool_tools['StoppingCriteria'] = []

    # Configure n-gram repetition prevention
    if dict_of_tools['no_repeat']:
        tool_tools['LogitsProcessors'] = [NoRepeatNGramLogitsProcessor(dict_of_tools['no_repeat'])]

    # Configure verse size enforcement
    if dict_of_tools['verse_size']:
        # First verse: enforce size from beginning
        if dict_of_tools['current_verse'] == 0:
            tool_tools['LogitsProcessors'].append(
                MinNewTokensLengthLogitsProcessor(dict_of_tools['input_length'],
                                                  dict_of_tools['verse_size'] + 1,  # +1 for EOS token
                                                  tokenizer.eos_token_id))
            tool_tools['LogitsProcessors'].append(
                ForcedEOSTokenLogitsProcessor(dict_of_tools['verse_size'] + 1,
                                              tokenizer.eos_token_id))
        # Subsequent verses: account for cumulative input length
        else:
            tool_tools['LogitsProcessors'].append(
                MinNewTokensLengthLogitsProcessor(dict_of_tools['input_length'],
                                                  dict_of_tools['verse_size'] + dict_of_tools['input_length'] + 1,
                                                  tokenizer.eos_token_id))
            tool_tools['LogitsProcessors'].append(
                ForcedEOSTokenLogitsProcessor(dict_of_tools['verse_size'] + dict_of_tools['input_length'] + 1,
                                              tokenizer.eos_token_id))
    else:
        # No verse size specified: enforce minimum length for non-first verses
        if dict_of_tools['current_verse'] != 0:
            tool_tools['LogitsProcessors'].append(
                MinNewTokensLengthLogitsProcessor(dict_of_tools['input_length'],
                                                  2,  # Minimum 2 tokens
                                                  tokenizer.eos_token_id))

    # Add semantic similarity wave modulation if configured
    if dict_of_tools['cos_sim']:
        tool_tools['LogitsProcessors'].append(
            MasterLogits(dict_of_tools, vocab, tokenizer))

    # Configure EOS token stopping criteria
    tool_tools['StoppingCriteria'].append(
        EndCriteria(tokenizer.eos_token_id))

    return tool_tools


def generator(prompt, active_tools, tokenizer, vocab, model, stream=False):
    """
    Generates multi-verse poetry using beam search with custom logits processors.

    Verses are generated sequentially, with each verse continuing from the previous one.
    The generation process applies semantic similarity wave modulation and other
    constraints specified in active_tools.

    Args:
        prompt (str): Initial text to start the poem generation.
        active_tools (dict): Configuration dictionary containing generation parameters.
        tokenizer: Tokenizer for encoding/decoding text.
        vocab (dict): Complete vocabulary dictionary.
        model: GPT-2 language model for text generation.
        stream (bool): If True, yields verses as they're generated. If False, returns complete poem dict.

    Returns/Yields:
        If stream=False: dict with keys 'VERSE {n}' containing decoded text and 'VERSE {n} tokens' containing token IDs
        If stream=True: yields tuples of (verse_num, verse_text, verse_tokens) for each verse
    """
    # Beam search configuration
    input_text = prompt
    poem = {}

    for num_verse in range(active_tools['num_verses']):
        # Update verse tracking for context-aware processors
        active_tools['current_verse'] = num_verse

        # Tokenize current input
        prompt_tokenized = tokenizer(input_text, return_tensors='pt')['input_ids']

        # Configure logits processors and stopping criteria
        tool_tools = get_tool_tools(active_tools, tokenizer, vocab)
        logits_processor = LogitsProcessorList(tool_tools['LogitsProcessors'])
        stopping_criteria = StoppingCriteriaList(tool_tools['StoppingCriteria'])

        max_length = active_tools['input_length'] + active_tools['verse_size']
        generated = model.generate(
            prompt_tokenized,
            max_length=max_length,
            num_beams=NUM_BEAMS,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=PAD_TOKEN_ID
        )

        # Store all beam predictions
        predictions = generated

        # Truncate at EOS token if present
        try:
            clean_predictions = [beam.tolist()[0:beam.tolist().index(EOS_TOKEN_ID)] for beam in predictions]
        except ValueError:
            clean_predictions = [beam.tolist() for beam in predictions]

        # Select best beam (index 0)
        verse_tokens = clean_predictions[0]

        # Prepare input for next verse by removing EOS token
        try:
            input_text = tokenizer.decode(verse_tokens[0:verse_tokens.index(EOS_TOKEN_ID)])
        except ValueError:
            input_text = tokenizer.decode(verse_tokens)

        # Update input length for next iteration
        active_tools['input_length'] = len(verse_tokens)

        # Decode verse text (exclude original prompt for subsequent verses)
        if num_verse == 0:
            verse = tokenizer.decode(verse_tokens)
        else:
            verse = tokenizer.decode(verse_tokens[len(prompt_tokenized[0]):])

        # Store verse text and tokens
        poem[f'VERSE {num_verse}'] = verse
        poem[f'VERSE {num_verse} tokens'] = verse_tokens

        # If streaming, yield the verse immediately
        if stream:
            yield (num_verse, verse, verse_tokens)

    # If not streaming, return complete poem dict
    if not stream:
        return poem