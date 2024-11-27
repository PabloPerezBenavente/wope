import pronouncing
import tensorflow as tf
import numpy as np
import torch
from numpy.linalg import norm

def get_rhyme_and_syl_data(rhyme_to_numeric_tokens, numeric_tokens_to_syl):
  rhyme_and_syl_dict = {}
  no_rhyme_and_syl_dict = {}
  for rhyming_part in rhyme_to_numeric_tokens.keys():
    rhyming_part_with_syl = {rhyming_part: {}}
    for token in rhyme_to_numeric_tokens[rhyming_part]:
      try:
        rhyming_part_with_syl[rhyming_part][numeric_tokens_to_syl[token]].append(token)
      except KeyError:
        try:
          rhyming_part_with_syl[rhyming_part][numeric_tokens_to_syl[token]] = [token]
        except KeyError:
          pass
    rhyme_and_syl_dict.update(rhyming_part_with_syl)
    words_that_rhyme = 0
    for sylable in rhyme_and_syl_dict[rhyming_part]:
      for word in rhyme_and_syl_dict[rhyming_part][sylable]:
        words_that_rhyme += 1
      if words_that_rhyme == 1:
        try:
          no_rhyme_and_syl_dict[sylable].extend(rhyme_and_syl_dict[rhyming_part][sylable])
        except KeyError:
          no_rhyme_and_syl_dict[sylable] = [rhyme_and_syl_dict[rhyming_part][sylable]]

  return rhyme_and_syl_dict, no_rhyme_and_syl_dict

def get_embedding(word, vocab, model):
  """
  Args: word (str)
  returns: embedding
  """

  id = vocab[word]
  embedding = model.transformer.wte.weight[id]

  return embedding


def get_similarity(key, query, vocab, model):

  if type(key) == str:
    key = get_embedding(key, vocab, model).detach().numpy()
  if type(query) == str:
    query = get_embedding(query, vocab, model).detach().numpy()
  cosine_similarity = np.dot(query, key)/(norm(query)*norm(key))

  return cosine_similarity


def create_boosted_vocab(ids, value, vocab):
  """
  Args: ids(list) a list with the ids of tokens to boost
        value (int) the value to boost desired tokens
  returns: boosted_vocab: a tensor with the shape of the vocabulary, where only boosted tokens have positive constant value
  """

  ids_to_boost = tf.cast(tf.reshape(tf.convert_to_tensor(ids), [len(ids), 1]), tf.int64)
  values_tensor = tf.cast(tf.fill([len(ids)], int(value)), tf.float32)
  shape = [len(vocab)]

  boosted_vocab = tf.sparse.SparseTensor(ids_to_boost, values_tensor, shape)

  return boosted_vocab


def get_word_tokens_not_in_cmu(vocab):

  word_tokens_in_cmu = []
  word_tokens_not_in_cmu = []


  for word in vocab.keys():
    in_cmu = False
    if pronouncing.phones_for_word(word.strip(' Ġ')):
      for character in word:
        if character != 'I' and character != 'Ġ' and character.isupper():
          in_cmu = False
          break
        else:
          in_cmu = True
    if in_cmu:
      word_tokens_in_cmu.append(word)
    else:
      word_tokens_not_in_cmu.append(word)



  return word_tokens_not_in_cmu, word_tokens_in_cmu


def rhymes_to_numeric_tokens(vocab_in_cmu, vocab):

  """
  args: list with the gpt2 tokens that are in cmu dict when .strip('Ġ')
  returns: dict with rhyming strings as keys and lists of the words with that rhyming part as values
  """

  rhyme_dict = {}
  tokens_that_dont_rhyme = {}
  tokens_with_polisylabic_rhyme = {}

  for token in vocab_in_cmu:
    flag = False
    if token == 'money':
      pass
      #print('analysing money')
      #flag = True
    if token == 'Ġmoney':
      pass
      #print('analysing Ġmoney')
      #flag = True

    # create dictionary of non-rhyming tokens sorted by syllables:

    # first add tokens with no rhymes
    if pronouncing.rhymes(token.strip(' Ġ')) == []:
      if flag == True:
        pass
        #print('token didnt have rhymes')

      try:
        tokens_that_dont_rhyme[pronouncing.syllable_count(pronouncing.phones_for_word(token.strip(' Ġ'))[0])].append(vocab[token])
      except KeyError:
        tokens_that_dont_rhyme[pronouncing.syllable_count(pronouncing.phones_for_word(token.strip(' Ġ'))[0])] = [vocab[token]]

    # then add tokens whose rhymes are not in vocab
    elif pronouncing.rhymes(token.strip(' Ġ')) != []:
      rhymes_in_vocab = True

      for rhyme in pronouncing.rhymes(token.strip(' Ġ')):
        if rhyme not in vocab_in_cmu and 'Ġ' + rhyme not in vocab_in_cmu:
          rhymes_in_vocab = False
        else:
          rhymes_in_vocab = True
          break
      if rhymes_in_vocab == False:
        if flag == True:
          pass
          #print('rhymes of token were not in vocab, so token is treated as if it didnt rhyme')
        #print(f"token {token} has rhymes but none of them are in vocab")
        try:
          tokens_that_dont_rhyme[pronouncing.syllable_count(pronouncing.phones_for_word(token.strip(' Ġ'))[0])].append(vocab[token])
        except KeyError:
          tokens_that_dont_rhyme[pronouncing.syllable_count(pronouncing.phones_for_word(token.strip(' Ġ'))[0])] = [vocab[token]]
      else:
        if flag == True:
          pass
          #print('rhymes were in vocab')

        if len(pronouncing.stresses(pronouncing.phones_for_word(token.strip(' Ġ'))[0])) > 1 and pronouncing.stresses(pronouncing.phones_for_word(token.strip('Ġ'))[0])[-1] != 1:
          if flag == True:
            pass
            #print('token rhyme is more than one syllable')
          try:
            tokens_with_polisylabic_rhyme[pronouncing.syllable_count(pronouncing.phones_for_word(token.strip(' Ġ'))[0])].append(vocab[token])
          except KeyError:
            tokens_with_polisylabic_rhyme[pronouncing.syllable_count(pronouncing.phones_for_word(token.strip(' Ġ'))[0])] = [vocab[token]]
        else:
          if flag == True:
            pass
            #print('token rhyme is only one syllable, so it is included')
          try:
            rhyme_dict[pronouncing.rhyming_part(pronouncing.phones_for_word(token.strip(' Ġ'))[0])].append(vocab[token])
          except KeyError:
            rhyme_dict[pronouncing.rhyming_part(pronouncing.phones_for_word(token.strip(' Ġ'))[0])] = [vocab[token]]

    #elif len(pronouncing.stresses(pronouncing.phones_for_word(token.strip(' Ġ'))[0])) > 1 and pronouncing.stresses(pronouncing.phones_for_word(token.strip('Ġ'))[0])[-1] != 1:
      #if token.strip('Ġ') == 'important':
        #print(token)
      #if flag == True:
        #print('token rhyme is more than one syllable')
      #try:
        #tokens_with_polisylabic_rhyme[pronouncing.syllable_count(pronouncing.phones_for_word(token.strip(' Ġ'))[0])].append(vocab[token])
      #except KeyError:
        #tokens_with_polisylabic_rhyme[pronouncing.syllable_count(pronouncing.phones_for_word(token.strip(' Ġ'))[0])] = [vocab[token]]

  return rhyme_dict, tokens_that_dont_rhyme, tokens_with_polisylabic_rhyme


def get_syl_items(word_tokens_in_cmu, vocab):

  numeric_tokens_to_syl = {}
  syl_to_numeric_tokens = {}

  for token in word_tokens_in_cmu:
    clean_token = token.strip(' Ġ')
    for i in range(6):
      if pronouncing.syllable_count(pronouncing.phones_for_word(clean_token)[0]) == i+1:
        numeric_tokens_to_syl[vocab[token]] = i+1
        try:
          syl_to_numeric_tokens[i+1].append(vocab[token])
        except KeyError:
          syl_to_numeric_tokens[i+1] = [vocab[token]]

  return numeric_tokens_to_syl, syl_to_numeric_tokens

def get_semantic_items(word, vocab, model):

  """
  args:
  cos_sim: a word (str)
  returns:
  list of tokens that are close
  """

  similar_words = []
  for n, token in enumerate(vocab):
    query = token

    # check that token is in cmu, to get meaningfull words?
    if pronouncing.phones_for_word(token.strip(' Ġ').lower()):
      if token.strip(' Ġ').lower() not in word and word not in token.strip(' Ġ').lower():
          key = word
          similarity = get_similarity(query, key, vocab, model)
          similar_words.append((token, similarity))

  # filter only the top nth closest
  ordered_words = sorted(similar_words, key = lambda x: x[1], reverse=True)

  #print('similar words', ordered_words[:50])
  similar_tokens = [vocab[token[0]] for token in ordered_words[:50]]

  return similar_tokens


def ban_scores_syl(tokenizer, beam_input_ids, numeric_tokens_to_syl, syl_to_numeric_tokens, length_of_verse_in_syllables,
                   rhyme=False, rhyme_type=False, current_verse=False):
  banned_tokens_in_beam = []
  boosted_tokens_in_beam = []
  # print('STARTING SYLLABLE PROCESSING')

  # set syllable counter to zero
  syl_count = 0

  for token in beam_input_ids.tolist():
    if tokenizer.decode(token)[0] != "'" and tokenizer.decode(token) != "s":
      try:
        syl_count += numeric_tokens_to_syl[token]
      except KeyError:
        pass

  # print(f"current sylables are {syl_count}")
  for syl in syl_to_numeric_tokens:
    # ban words that would go over the max number of syllables.
    if syl_count + syl > length_of_verse_in_syllables:
      # print(f'banning words with {syl} syllables')
      banned_tokens_in_beam.extend([token for token in syl_to_numeric_tokens[syl]])

    # if rhyme is hard
    if rhyme_type == 'hard':

      # if a word ends the verse
      if syl_count + syl == length_of_verse_in_syllables:

        # get rhyming part according to rhyme type of current verse
        rhyme_type = rhyme['list_scheme'][current_verse]
        current_rhyming_part = rhyme['rhyme_scheme'][rhyme_type]['rhyming_part']

        # if this is the first verse of its type and it must rhyme, then
        # ban words with no rhymes
        if current_rhyming_part == [] and rhyme['list_scheme'].count(rhyme_type) != 1:
          # print(f"elliminating tokens that dont rhyme and have {syl} syllables")
          try:
            try:
              banned_tokens_in_beam.extend(rhyme['no_rhyme_syl_dict'][syl])
            except KeyError:
              pass
          # except KeyError:
          # pass
          except TypeError:
            try:
              banned_tokens_in_beam.append(rhyme['no_rhyme_syl_dict'][syl])
            except KeyError:
              pass

          # ban words that dont have rhymes with one syllable:
          # print(f"elliminating tokens that dont rhyme with monosyllables and have {syl} syllables")
          try:
            banned_tokens_in_beam.extend(rhyme['polisylabic_rhyme'][syl])
          except KeyError:
            pass
          except TypeError:
            banned_tokens_in_beam.append(rhyme['polisylabic_rhyme'][syl])



        # if it is not the first prediction for that rhyme type
        else:
          # print('current verse must rhyme with ', current_rhyming_part)

          # ban tokens that will finish verse and dont rhyme:

          # print(f"banning tokens that dont rhyme and are {syl} syllables long")

          banned_tokens_in_beam.extend(rhyme['no_rhyme_stl_dict'][syl])

          # ban tokens that will finish verse and dont rhyme with current rhyming part

          for rhyming_part in rhyme['rhyme_and_syl_dict']:
            if rhyming_part != current_rhyming_part:
              # if rhyming_part == 'IH1 L M':
              # print('analyzing film rhyming part')
              try:

                tokens_that_wouldnt_rhyme = rhyme['rhyme_and_syl_dict'][rhyming_part][syl]
                banned_tokens_in_beam.extend(tokens_that_wouldnt_rhyme)

                # print(f"tokens ending in {rhyming_part} are: ", tokens_that_wouldnt_rhyme)
                # if rhyming_part == 'IH1 L M':
                # print('banned tokens rhyming with film: ', tokens_that_wouldnt_rhyme)

                # if 45885 in tokens_that_wouldnt_rhyme:
                # print(f'Ġcrotch has been banned')
                # if 1169 in tokens_that_wouldnt_rhyme:
                # print(f'Ġthe has been banned')
                # if 262 in tokens_that_wouldnt_rhyme:
                # print(f'the has been banned')
              except KeyError:
                pass

          # also ban words that have already been emitted, even if they have the same rhyming part:
          try:

            # print('banning words that were already rhymed')
            rhyming_words = rhyme['rhyme_scheme'][rhyme_type]['rhyming_words']
            # print('already emitted rhyiming words: ', rhyming_words)
            # print(f"banning {rhyming_words}")
            banned_tokens_in_beam.extend(rhyming_words)


          except TypeError:
            banned_tokens_in_beam.append(rhyme['rhyme_scheme'][rhyme_type]['rhyming_words'])

          try:
            boosted_tokens_in_beam.extend(
              [token for token in rhyme['rhyme_and_syl_dict'][current_rhyming_part][syl] if token not in rhyming_words])

          except KeyError:
            pass
            # print(f"no tokens can be boosted with {syl} syllables")

          except TypeError:
            for candidate_rhyming_word in rhyme['rhyme_and_syl_dict'][current_rhyming_part][syl]:
              try:
                if candidate_rhyming_word not in rhyming_words:
                  boosted_tokens_in_beam.append(candidate_rhyming_word)
              except TypeError:
                if candidate_rhyming_word != rhyming_words:
                  boosted_tokens_in_beam.append(candidate_rhyming_word)
  # if in banned_tokens_in_beam:
  # print('we banned 35505')

  # print('length of banned tokens in this beam: ', len(banned_tokens_in_beam))
  # print(f"this verse is {syl_count} syllables long")
  # if 45885 in banned_tokens_in_beam:
  # print(f'Ġcrotch has been banned when having {syl_count} syllables')
  # if 1169 in banned_tokens_in_beam:
  # print(f'Ġthe has been banned when having {syl_count} syllables')
  # if 262 in banned_tokens_in_beam:
  # print(f'the has been banned when having {syl_count} syllables')
  return banned_tokens_in_beam, boosted_tokens_in_beam

def ban_scores(scores, banned_tokens):


    #print('type of banned tokens: ', type(banned_tokens))
    #print('type of first element in banned tokens', type(banned_tokens[0]))
    """
    Args:
        scores: logits distribution of shape (batch size, vocabulary size)
        banned_tokens: list of list of tokens to ban of shape (batch_size, number_of_banned_tokens)

    returns: scores: tensor where banned token positions are set to '-inf'
    """

    # print('number of beams in tokens to ban: ', len(banned_tokens))
    # print('size of scores', scores.size())

    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
      for token in batch_banned_tokens:
        banned_mask_list.append([idx, token])
    if not banned_mask_list:
      return scores

    print(type(scores))

    #print('BANNED_MASK_LIST: ', type(banned_mask_list))
    #print('first element: ', type(banned_mask_list[0]))
    #print('first element of first element: ', type(banned_mask_list[0][0]))
    #print('first of first: ', banned_mask_list[0][0])
    try:
      banned_mask = torch.LongTensor(banned_mask_list)
    except TypeError:
      pass
      #print(banned_mask_list)
    indices = torch.ones(len(banned_mask))

    banned_mask = (
      torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    )

    scores = scores.masked_fill(banned_mask, -float("inf"))

    return scores

def boost_scores(scores, cos_sim_dict, tokens_to_boost, vocab):
  """
  Args: scores
        cos_sim_dict {'abcds':{'intensity':x, 'frequency':y, 'vocab':[]}}
        tokens_to_boost [[[beam1keyword1][beam1keyword2]][[beam2keyword1][beam2keyword2]]]
  returns: boosted_scores: a tensor with the modified scores
  """

  boosted_scores = []

  if cos_sim_dict:
    keyword_list = []
    for keyword in cos_sim_dict.keys():
      keyword_list.append(keyword)

    for beam in tokens_to_boost:


        new_beam = []

        #print('length of boosted beam tokens', len(beam))

        for n, topic_tokens in enumerate(beam):

          # create a tensor with value in the place of their correspondent vocab tokens.
          topic = keyword_list[n]
          ids_to_boost = tf.cast(tf.reshape(tf.convert_to_tensor(topic_tokens), [len(topic_tokens), 1]), tf.int64)
          values_tensor = tf.cast(tf.fill([len(topic_tokens)], cos_sim_dict[topic]['intensity']), tf.float32)
          shape = [len(vocab)]
          boosted_score = tf.sparse.SparseTensor(ids_to_boost, values_tensor, shape)

          #print(f"boosting {keyword_list[n]} with an intensity of {cos_sim_dict[topic]['intensity']}")

          # i dont know what this does
          boosted_score = tf.sparse.reorder(boosted_score)

          # transform to dense
          dense_vocab = tf.sparse.to_dense(boosted_score)

          # reshape to [1, vocabsize] to match each score's beam shape
          resized_dense_vocab = tf.expand_dims(dense_vocab, axis = 0)

          # transform to numpy
          beam_keyword_boost = resized_dense_vocab.numpy()

          # add the boosts to the beam
          if type(new_beam) == list:
            new_beam = beam_keyword_boost
          else:
            new_beam += beam_keyword_boost

        boosted_scores.append(new_beam)

    boosted_scores = np.array(boosted_scores)

    boosted_scores = np.squeeze(boosted_scores, axis = (1,))


  else:

    #boosted_scores = []

    #print('boosting scores without cos_sim_dict')

    #print('number of beams in tokens to boost with no cos sim: ', len(tokens_to_boost))
    for beam in tokens_to_boost:


        new_beam = []

        #for n, topic_tokens in enumerate(beam):

        # create a tensor with value in the place of their correspondent vocab tokens.
        #topic = keyword_list[n]
        ids_to_boost = tf.cast(tf.reshape(tf.convert_to_tensor(beam), [len(beam), 1]), tf.int64)
        values_tensor = tf.cast(tf.fill([len(beam)], 6), tf.float32)
        shape = [len(vocab)]
        boosted_score = tf.sparse.SparseTensor(ids_to_boost, values_tensor, shape)

        #print(f"boosting {keyword_list[n]} with an intensity of {cos_sim_dict[topic]['intensity']}")

        # i dont know what this does
        boosted_score = tf.sparse.reorder(boosted_score)

        # transform to dense
        dense_vocab = tf.sparse.to_dense(boosted_score)

        # reshape to [1, vocabsize] to match each score's beam shape
        resized_dense_vocab = tf.expand_dims(dense_vocab, axis = 0)

        # transform to numpy
        beam_boost = resized_dense_vocab.numpy()

        # add the boosts to the beam
        if type(new_beam) == list:
          new_beam = beam_boost
        else:
          new_beam += beam_boost


        # add beam with boosted values to the scores
        boosted_scores.append(new_beam)

    # transform list of arrays into array
    boosted_scores = np.array(boosted_scores)

    #if not cos_sim_dict:
    # reduce dimension corresponding to original list to match scores shape
    boosted_scores = np.squeeze(boosted_scores, axis=(1,))

  # operate as arrays and transform back again to tensor
  numpy_scores = scores.detach().numpy()
  new_scores = numpy_scores + boosted_scores
  boosted_scores = torch.tensor(new_scores)




  return boosted_scores

def get_syllables_in_verse(verse):
  syl = 0
  for word in verse.split(' '):
    try:
      syl += pronouncing.syllable_count(pronouncing.phones_for_word(word)[0])
    except IndexError:
      print('the error ocurred with: ', word)

  return syl