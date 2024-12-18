import pronouncing
import tensorflow as tf
import numpy as np
import torch
from numpy.linalg import norm

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

    banned_mask = torch.LongTensor(banned_mask_list)

    indices = torch.ones(len(banned_mask))

    banned_mask = (
      torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    )

    scores = scores.masked_fill(banned_mask, -float("inf"))

    return scores

def boost_scores(scores, cos_sim_dict, tokens_to_boost, vocab):
  """
  Args: scores
        cos_sim_dict {'abcds':{'intensity':x, 'period':y, 'vocab':[]}}
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
        for n, topic_tokens in enumerate(beam):

          # create a tensor with value in the place of their correspondent vocab tokens.
          topic = keyword_list[n]
          ids_to_boost = tf.cast(tf.reshape(tf.convert_to_tensor(topic_tokens), [len(topic_tokens), 1]), tf.int64)
          values_tensor = tf.cast(tf.fill([len(topic_tokens)], cos_sim_dict[topic]['intensity']), tf.float32)
          shape = [len(vocab)]
          boosted_score = tf.sparse.SparseTensor(ids_to_boost, values_tensor, shape)
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

  # operate as arrays and transform back again to tensor
  numpy_scores = scores.detach().numpy()
  new_scores = numpy_scores + boosted_scores
  boosted_scores = torch.tensor(new_scores)

  return boosted_scores
