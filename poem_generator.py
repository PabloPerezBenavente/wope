import pronouncing
import torch

from transformers import BeamSearchScorer, \
    LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria

from toolprocessors import get_tool_tools
from utils import get_syllables_in_verse

def generator(prompt, active_tools, tokenizer, vocab, model):

    poem = {}


    # how many beams to track during the Viterbi algorithm
    num_beams = 10
    # how many beams to return after the algorithm
    num_return_beams = 5

    # get length of verses
    fixed_length = active_tools['verse_size']
    prompt_length = active_tools['input_length']

    input = prompt

    print('the number of verses will be: ', active_tools['num_verses'])

    for num_verse in range(active_tools['num_verses']):

        print('generating verse', num_verse)

        # save current verse
        active_tools['current_verse'] = num_verse

        # the prompt to continue
        #input = prompt
        verse_prompt = input

        print(f'this will be the prompt of verse {num_verse}: ', verse_prompt)

        # tokenizing the prompt
        prompt_tokenized = tokenizer(verse_prompt, return_tensors='pt' )
        prompt_tokenized = prompt_tokenized['input_ids']

        # get instances for LogitsProcessors
        tool_tools = get_tool_tools(active_tools, tokenizer, vocab)

        # instantiating a BeamSearchScorer
        beam_scorer = BeamSearchScorer(
            batch_size = prompt_tokenized.shape[0],
            num_beams = num_beams,
            num_beam_hyps_to_keep = num_return_beams,
            device=model.device
        )

        # creating a list of LogitsProcessor instances
        logits_processor = LogitsProcessorList(tool_tools['LogitsProcessors'])

        # creating a list of StoppingCriteria instances
        stopping_criteria = StoppingCriteriaList(tool_tools['StoppingCriteria'])

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

        print(f'these are the raw predictions: {predictions}')

        clean_predictions = [beam.tolist()[0:beam.tolist().index(50256)] for beam in predictions]

        print(f'these are the predictions after cleaning: {clean_predictions}')

        # decode predictions
        decoded_predictions = [tokenizer.decode(beam) for beam in clean_predictions]

        #decoded_predictions = [tokenizer.decode(beam.tolist()[0:beam.tolist().index(50256)]) for beam in predictions]

        print('these are the decoded predictions: ', decoded_predictions)

        # BEAM POSTPROCESSING

        filtered_predictions = []

        # if syllabic length is active, get beams with appropriate number
        if active_tools['num_syl']['active']:

            # get appropriate number
            if num_verse == 0:
                try:
                    max_size = active_tools['num_syl']['scheme']['num_verse']
                except KeyError:
                    max_size = active_tools['num_syl']['number']
            else:
                try:
                    max_size = active_tools['num_syl']['input_syl'] + active_tools['num_syl']['scheme'][num_verse]
                except KeyError:
                    max_size = active_tools['num_syl']['number']*(num_verse+1)

            # select only beams with the appropriate number of syllables
            filtered_predictions = []
            for verse in predictions:
                decoded_verse = tokenizer.decode(verse.tolist()[0:verse.tolist().index(50256)])
                number_of_syllables = get_syllables_in_verse(decoded_verse)
                if number_of_syllables == max_size:
                    filtered_predictions.append(verse)



        # if rhyme is active
        if active_tools['rhyme']['active'] and active_tools['rhyme']['type'] == 'hard':
            rhyming_type = [active_tools['rhyme']['list_scheme'][num_verse]][0]
            if active_tools['rhyme']['rhyme_scheme'][rhyming_type]['rhyming_words'] != []:
                forcing_rhyme = True
                rhyming_part = active_tools['rhyme']['rhyme_scheme'][rhyming_type]['rhyming_part']

                beams_that_rhyme = []
                if active_tools['num_syl']['active']:
                    beams = filtered_predictions
                else:
                    beams = decoded_predictions

                try:
                    for n, beam in enumerate(beams):
                        end_index = beam.tolist().index(50256)

                        last_token = beam[end_index - 1]
                        last_token_rhyme = pronouncing.rhyming_part(pronouncing.phones_for_word(tokenizer.decode(last_token).strip(' Ġ'))[0])
                        if last_token_rhyme == rhyming_part:

                            beams_that_rhyme.append(beam)
                except TypeError:
                    failed_poem = 'this didnt work'
                    return failed_poem
            else:
                forcing_rhyme = False
        else:
            forcing_rhyme = False

        if forcing_rhyme:
            beams = beams_that_rhyme
            try:
                if (active_tools['num_syl'] and not active_tools['rhyme']) or (active_tools['num_syl'] and active_tools['rhyme']['rhyme_scheme'][rhyming_type]['rhyming_words'] == []):
                    beams = filtered_predictions
            except KeyError:
                pass

        if not forcing_rhyme and not active_tools['num_syl']['active']:
            good_beam = clean_predictions[0]

        elif not forcing_rhyme and active_tools['num_syl']['active']:
            good_beam = filtered_predictions[0]

        else:

            try:
                good_beam = beams[0]
            except IndexError:
                print('no poem could be generated')
                failed_poem = 'this didnt work'

            return failed_poem

        # eliminate <|endoftext|>
        output = good_beam
        #print('this is the output: ', output)


        verse_tokens = good_beam

        # construct verse
        #full_verse = ''
        #for n, token in enumerate(verse_tokens):
            #clean_token = tokenizer.decode(token)
            #try:
                #if clean_token not in ['er', 'ory', 'ing', 's', 'na', 'res', 'es', 'ate', 'ed'] and clean_token[0] != " " and clean_token[0] not in ",!);':." and tokenizer.decode(verse_tokens[n-1]) != "(":
                    #full_verse += " "
            #except IndexError:
                #pass
            #full_verse += clean_token
        full_verse = tokenizer.decode(verse_tokens)

        # if rhyme is active and this is the first verse, prepare all tokens that dont rhyme
        if active_tools['rhyme']['active']:

            # if rhyme is hard, we save the rhyming part of the last word in its slot
            if active_tools['rhyme']['rype'] == 'hard':

                # get last word
                last_token = tokenizer.decode(verse_tokens[-1])
                last_word = last_token.strip(' Ġ')

                # get rhyming part
                try:
                    rhyming_part = pronouncing.rhyming_part(pronouncing.phones_for_word(last_word)[0])
                except IndexError:
                    pass

                # if rhyme is hard, we save info of previous word according to scheme
                if active_tools['rhyme']['type'] == 'hard':

                    # get rhyme type of current verse (AABB):
                    rhyme_type = active_tools['rhyme']['list_scheme'][num_verse]

                    # save rhyming part and word in their slots
                    active_tools['rhyme']['rhyming_scheme'][rhyme_type].update({'rhyming_part' : rhyming_part})
                    active_tools['rhyme']['rhyming_scheme'][rhyme_type]['rhyming_words'].append(verse_tokens[-1])

                    if last_token[0] == ' ':

                        try:
                            active_tools['rhyme']['rhyme_scheme'][rhyme_type]['rhyming_words'].append(vocab[last_word])
                        except KeyError:
                            pass

                    else:
                        try:
                            active_tools['rhyme']['rhyme_scheme'][rhyme_type]['rhyming_words'].append(vocab['Ġ' + last_word])
                        except KeyError:
                            pass

                else:
                    pass

        input = full_verse
        print('this is set as the input: ', input)

        # set length of input for next verse
        active_tools['input_length'] = len(verse_tokens)

        # set length of input in syllables for next verse
        if active_tools['num_syl']['active']:
            try:
                if num_verse == 0:
                    try:
                        active_tools['num_syl']['input_syl'] = active_tools['num_syl']['scheme'][num_verse]
                    except KeyError:
                        active_tools['num_syl']['inpu_syl'] = active_tools['num_syl']['number']
                else:
                    try:
                        active_tools['num_syl']['input_syl'] = active_tools['num_syl']['scheme'][num_verse] + active_tools['num_syl']['input_syl']
                    except KeyError:
                        active_tools['num_syl']['input_syl'] = active_tools['num_syl']['number']*(num_verse+1)
            except IndexError:
                pass

        if num_verse == 0:
            verse = tokenizer.decode(verse_tokens)
        else:
            verse = tokenizer.decode(verse_tokens[len(prompt_tokenized[0]):])

        poem[f'VERSE {num_verse}'] = verse
        poem[f'VERSE {num_verse} tokens'] = verse_tokens
        #if num_verse != 0:
            #poem[f'VERSE {num_verse} tokens'] = output.tolist()[len(prompt_tokenized[0]):]
        #else:
            #poem[f'VERSE {num_verse} tokens'] = output.tolist()

        print(verse)

    return poem