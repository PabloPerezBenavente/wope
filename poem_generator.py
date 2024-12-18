import torch
from transformers import BeamSearchScorer, \
    LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria

from toolprocessors import get_tool_tools

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

        # Set up a beam search scorer to manage beam search, i.e., finding the best poetic lines
        beam_scorer = BeamSearchScorer(
            batch_size = prompt_tokenized.shape[0], # How many input examples we're scoring (just one here)
            num_beams = num_beams, # Number of beams we're scoring per input
            num_beam_hyps_to_keep = num_return_beams, # Keep the top n hypotheses
            device=model.device # Run everything on the same device as the model
        )

        # Perform beam search to generate poetic output with controlled creativity
        generated = model.beam_search(
            torch.cat([prompt_tokenized] * num_beams), # Duplicate input for each beam
            beam_scorer, # Use the scorer defined above
            logits_processor=logits_processor, # Apply custom constraints during generation
            pad_token_id=0, # Pad token to avoid crashes with short outputs
            stopping_criteria=stopping_criteria # Stop when conditions are met (e.g., verse ends)
        )

        # Save the raw predictions (all beams)
        predictions = generated

        # Clean up predictions by truncating tokens at <|endoftext|> (50256 is the magic token ID for that)
        clean_predictions = [beam.tolist()[0:beam.tolist().index(50256)] for beam in predictions]

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