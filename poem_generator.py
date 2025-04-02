from transformers import LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria
from transformers import NoRepeatNGramLogitsProcessor, MinNewTokensLengthLogitsProcessor, ForcedEOSTokenLogitsProcessor
from masterlogits import MasterLogits
from endcriteria import EndCriteria


def get_tool_tools(dict_of_tools, tokenizer, vocab):
    """
    Loads all the generation tools (logit processors and stopping criteria) into one neat package.
    Think of it as the toolbox that holds everything you need to manipulate, restrict, and stop generation.
    """
    tool_tools = {}

    # Number of verses we want (because poems need limits, unlike your imagination)
    tool_tools['num_verses'] = dict_of_tools['num_verses']

    # Create a slot for all the LogitsProcessors (fancy tools for controlling the model's word choices)
    tool_tools['LogitsProcessors'] = []

    # Create a slot for StoppingCriteria (the emergency brake for generation)
    tool_tools['StoppingCriteria'] = []

    # Set repetition restrictions (because no one likes repetitive poetry)
    if dict_of_tools['no_repeat']:
        # Prevent n-grams from repeating themselves like a broken record
        tool_tools['LogitsProcessors'] = [NoRepeatNGramLogitsProcessor(dict_of_tools['no_repeat'])]

    # Enforce fixed verse size using min/max token length (poetry with rules is still poetry)
    if dict_of_tools['verse_size']:
        # Special case for the first verse (get ready to start strong!)
        if dict_of_tools['current_verse'] == 0:
            tool_tools['LogitsProcessors'].append(
                MinNewTokensLengthLogitsProcessor(dict_of_tools['input_length'],
                                                  dict_of_tools['verse_size'] + 1, # Add 1 for the <|endoftext|> token
                                                  tokenizer.eos_token_id))
            tool_tools['LogitsProcessors'].append(
                ForcedEOSTokenLogitsProcessor(dict_of_tools['verse_size'] + 1,
                                              tokenizer.eos_token_id))
        # For subsequent verses (because consistency matters... sometimes)
        else:
            tool_tools['LogitsProcessors'].append(
                MinNewTokensLengthLogitsProcessor(dict_of_tools['input_length'],
                                                  dict_of_tools['verse_size'] + dict_of_tools['input_length'] + 1,
                                                  tokenizer.eos_token_id))
            tool_tools['LogitsProcessors'].append(
                ForcedEOSTokenLogitsProcessor(dict_of_tools['verse_size'] + dict_of_tools['input_length'] + 1,
                                              tokenizer.eos_token_id))
    else:
        # If no verse size is specified and it's not the first verse, set a minimum token length
        if dict_of_tools['current_verse'] != 0:
            tool_tools['LogitsProcessors'].append(
                MinNewTokensLengthLogitsProcessor(dict_of_tools['input_length'],
                                                  2, # Minimum 2 tokens (because a verse with 1 word isn't impressive)
                                                  tokenizer.eos_token_id))

    # Apply semantic restrictions (aka the "poetry police")
    if dict_of_tools['cos_sim']:
        tool_tools['LogitsProcessors'].append(
            MasterLogits(dict_of_tools, vocab, tokenizer)) # The brains behind semantic similarity magic

    # Set stopping criteria (stop when <|endoftext|> shows up — because all things good come to an end)
    tool_tools['StoppingCriteria'].append(
        EndCriteria(tokenizer.eos_token_id))

    return tool_tools # Return the fully loaded toolbox, ready for action


def generator(prompt, active_tools, tokenizer, vocab, model):

    # Number of beams to track during the beam search algorithm (more beams = wider search space)
    num_beams = 10
    # Number of beams to return after the search
    num_return_beams = 5

    input = prompt # Start the poem with the initial prompt
    poem = {} # This dictionary will hold our poetic masterpiece

    for num_verse in range(active_tools['num_verses']):  # Generate multiple verses as specified
        # Keep track of which verse we're working on (important for tools that need context)
        active_tools['current_verse'] = num_verse

        # Convert the input prompt into token IDs the model can understand
        prompt_tokenized = tokenizer(input, return_tensors='pt' )['input_ids']

        # Get custom settings for logits processing and stopping criteria (for controlling generation rules)
        tool_tools = get_tool_tools(active_tools, tokenizer, vocab)
        logits_processor = LogitsProcessorList(tool_tools['LogitsProcessors'])
        stopping_criteria = StoppingCriteriaList(tool_tools['StoppingCriteria'])

        max_length = active_tools['input_length'] + active_tools['verse_size']
        generated = model.generate(
            prompt_tokenized,
            max_length=max_length,
            num_beams=num_beams,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=0
        )

        # Save the raw predictions (all beams)
        predictions = generated

        # Clean up predictions by truncating tokens at <|endoftext|> (50256 is the magic token ID for that)
        try:
            clean_predictions = [beam.tolist()[0:beam.tolist().index(50256)] for beam in predictions]
        except ValueError:
            clean_predictions = [beam.tolist() for beam in predictions]

        # Grab the best beam (the one with the highest score)
        verse_tokens = clean_predictions[0]

        # Remove <|endoftext|> for smooth continuation into the next verse
        try:
            input = tokenizer.decode(verse_tokens[0:verse_tokens.index(50256)])
        except ValueError as e: # If <|endoftext|> isn't found, decode all tokens
            input = tokenizer.decode(verse_tokens)

        # Update the length of the current input to track how far we've generated
        active_tools['input_length'] = len(verse_tokens)

        # Decode the tokens into human-readable poetry (exclude the prompt for subsequent verses)
        if num_verse == 0: # For the first verse, decode the entire thing
            verse = tokenizer.decode(verse_tokens)
        else: # For later verses, exclude the initial prompt tokens
            verse = tokenizer.decode(verse_tokens[len(prompt_tokenized[0]):])

        # Save the decoded verse and its tokenized version into the poem dictionary
        poem[f'VERSE {num_verse}'] = verse
        poem[f'VERSE {num_verse} tokens'] = verse_tokens

    # Return the final poetic creation, complete with tokens for each verse (for debugging or further processing)
    return poem