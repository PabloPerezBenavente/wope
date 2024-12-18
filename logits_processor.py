import torch
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from transformers import LogitsProcessor

class SemLogits(LogitsProcessor):
  def __init__(self, keyword, vocab, tokenizer, get_embedding, create_boosted_vocab):
    """
    this should take a word, create a tensor with the shape of the vocabulary where all tokens
    whose cosine similarity with that word is above a threshold have a positive value, and add
    this tensor to the outputted scores.
    """

    key = get_embedding(keyword).detach().numpy()

    # get list of tokens whose similarity w.r.t keyword is above threshold
    tokens_to_boost = []
    for token in vocab:
      query = get_embedding(token).detach().numpy()
      similarity = np.dot(query, key) / (norm(query) * norm(key))

      if similarity > 0.4 and keyword not in token.lower():
        tokens_to_boost.append(vocab[token])

    # remove keyword from list of tokens to boost
    for token in tokens_to_boost:
      if keyword in tokenizer.decode(token).strip(' Ġ').lower():
        tokens_to_boost.remove(token)

    # get vector of shape [vocab_size] where all tokens are 0 except for tokens to boost
    self.boosted_vocab = create_boosted_vocab(tokens_to_boost, float(8))

  def __call__(self, input_ids, scores):
    """
    args: input_ids
          scores
    returns: changed scores
    """

    # transform boosted vocabulary in a sparse tensor
    dense_vocab = tf.sparse.to_dense(self.boosted_vocab)

    # change shape to [1, vocab_size] to match scores shape
    resized_dense_vocab = tf.expand_dims(dense_vocab, axis=0)

    # for beam search
    scores = scores.detach().numpy()
    resized_dense_vocab.numpy()
    boosted_scores = scores + resized_dense_vocab

    boosted_scores = torch.tensor(boosted_scores.numpy())

    return boosted_scores