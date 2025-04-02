import pronouncing
import tensorflow as tf
import numpy as np
import torch
from numpy.linalg import norm

def get_embedding(word, vocab, model):
  """
    Encodes a word into its magical vector form, so it can enter the sacred world of embeddings.
    Args:
        word (str): The word to encode. Hopefully, it's in the vocabulary.
        vocab (dict): Maps words to their corresponding token IDs. The dictionary of dreams.
        model (object): The language model with pre-trained embeddings (aka the treasure chest).
    Returns:
        embedding (torch.Tensor): The numerical representation of the word, ready for deep learning shenanigans.
    """

  # Look up the word's ID in the vocabulary (no ID, no embedding — life's tough)
  id = vocab[word]

  # Fetch the word's embedding from the model's embedding layer
  embedding = model.transformer.wte.weight[id]

  return embedding


def get_similarity(key, query, vocab, model):
  """
    Calculates the cosmic closeness (cosine similarity) between two words or embeddings.
    Args:
        key (str or np.ndarray): The first word or its embedding. Can be a string or already a vector.
        query (str or np.ndarray): The second word or its embedding. Same deal as `key`.
        vocab (dict): Maps words to token IDs. Because words are useless without their IDs.
        model (object): The language model that holds the magical pre-trained embeddings.
    Returns:
        cosine_similarity (float): A number between -1 and 1 indicating how much the two inputs like each other.
    """
  # If the key is a word (string), encode it into an embedding (because math doesn't speak English)
  if type(key) == str:
    key = get_embedding(key, vocab, model).detach().numpy()
  # If the query is a word (string), encode it into an embedding as well (same reason as above)
  if type(query) == str:
    query = get_embedding(query, vocab, model).detach().numpy()

  # Calculate cosine similarity: dot product of the vectors divided by their magnitudes
  cosine_similarity = np.dot(query, key)/(norm(query)*norm(key))

  # Return their relationship status as a float
  return cosine_similarity


def create_boosted_vocab(ids, value, vocab):
  """
    Crafts a sparse tensor where chosen tokens shine brighter than the rest.
    Args:
        ids (list): A list of token IDs deemed worthy of a boost (the chosen ones).
        value (int): The constant value that will amplify the selected tokens' scores.
        vocab (dict): The complete vocabulary, serving as the canvas for this sparse tensor.
    Returns:
        boosted_vocab (tf.sparse.SparseTensor): A sparse tensor where the chosen tokens carry the boost,
        while the rest remain silent at zero.
    """
  # Reshape the IDs of the chosen tokens into a column vector (they’re ready for their spotlight)
  ids_to_boost = tf.cast(tf.reshape(tf.convert_to_tensor(ids), [len(ids), 1]), tf.int64)

  # Create a tensor filled with the boost value (a chorus of amplification)
  values_tensor = tf.cast(tf.fill([len(ids)], int(value)), tf.float32)

  # Define the shape of the sparse tensor to match the size of the vocabulary (the stage is set)
  shape = [len(vocab)]

  # Build a sparse tensor where only the chosen tokens receive their amplified value (the stars of the show)
  boosted_vocab = tf.sparse.SparseTensor(ids_to_boost, values_tensor, shape)

  # Return this sparse masterpiece, where the special tokens stand tall
  return boosted_vocab

def get_semantic_items(word, vocab, model):
    """
    Finds tokens in the vocab that are BFFs (most semantically similar) with the given word.
    Args:
        word (str): The lucky word we’re finding semantic buddies for.
        vocab (dict): The entire vocabulary.
        model (object): The magical neural net that helps us calculate similarity.
    Returns:
        similar_tokens (list): A list of token IDs representing the top 50 most similar words.
    """

    similar_words = []
    for n, token in enumerate(vocab):
        query = token # Current token we’re evaluating as a possible semantic pal.

        # Check if the token has a valid pronunciation (avoiding nonsense words)
        if pronouncing.phones_for_word(token.strip(' Ġ').lower()):
            # Make sure the token isn’t the same as or contains the input word (no narcissists or nested buddies)
            if token.strip(' Ġ').lower() not in word and word not in token.strip(' Ġ').lower():
              key = word # The input word becomes the "key" for comparison.
              similarity = get_similarity(query, key, vocab, model)
              similar_words.append((token, similarity)) # Add this token and its similarity score to the list.

    # Sort tokens by their similarity scores in descending order (because only the elite matter).
    ordered_words = sorted(similar_words, key = lambda x: x[1], reverse=True)

    # Take the top 50 tokens and convert them into their token IDs (the VIP list).
    similar_tokens = [vocab[token[0]] for token in ordered_words[:50]]

    return similar_tokens

def ban_scores(scores, banned_tokens):
    """
    Args:
        scores: logits distribution of shape (batch size, vocabulary size)
        banned_tokens: list of list of tokens to ban of shape (batch_size, number_of_banned_tokens)

    Returns:
        scores: tensor where banned token positions are set to '-inf'
    """

    # A list to hold coordinates of all the words we're sending to exile
    banned_mask_list = []

    for idx, batch_banned_tokens in enumerate(banned_tokens):
      for token in batch_banned_tokens:
          # Collecting (batch index, token index) pairs to banish them from the realm of scores
          banned_mask_list.append([idx, token])

    # If no tokens are banned, all is forgiven. Return original scores.
    if not banned_mask_list:
      return scores

    # Convert the exile list into a tensor of banned coordinates
    banned_mask = torch.LongTensor(banned_mask_list)

    # Create a tensor of ones (a symbolic representation of the act of banning)
    indices = torch.ones(len(banned_mask))

    # Build a sparse tensor to represent banned positions and make it dense and boolean
    banned_mask = (
      torch.sparse.LongTensor(banned_mask.t(), indices, scores.size())
      .to(scores.device)
      .to_dense()
      .bool() # True means: thou shalt not pass
    )

    # Mask out the banned tokens by setting their scores to '-inf' (eternal banishment)
    scores = scores.masked_fill(banned_mask, -float("inf"))

    return scores

def boost_scores(scores, cos_sim_dict, tokens_to_boost, vocab):
    """
      Args:
          scores: The logits distribution to tweak (shape: [batch size, vocab size])
          cos_sim_dict: {'keyword': {'intensity': x, 'period': y, 'vocab': []}}
          tokens_to_boost: [[[beam1keyword1][beam1keyword2]], [[beam2keyword1][beam2keyword2]]]
          vocab: The holy dictionary of tokens.

      Returns:
          boosted_scores: A tensor with the modified scores.
    """

    boosted_scores = []

    # Only boost if we’ve got a cos_sim_dict. Otherwise, no favoritism today.
    if cos_sim_dict:
        keyword_list = list(cos_sim_dict.keys()) # Extract all the keywords we care about boosting.

        for beam in tokens_to_boost: # Each beam gets its own boost treatment.
            new_beam = []
            for n, topic_tokens in enumerate(beam):
                # Step 1: Identify which tokens to give the "VIP treatment."
                topic = keyword_list[n] # Corresponding keyword for this set of tokens.
                ids_to_boost = tf.cast(
                    tf.reshape(tf.convert_to_tensor(topic_tokens), [len(topic_tokens), 1]), tf.int64
                                       ) # Get the IDs of tokens to boost.
                values_tensor = tf.cast(
                    tf.fill([len(topic_tokens)], cos_sim_dict[topic]['intensity']), tf.float32
                ) # Assign the intensity of favoritism.
                shape = [len(vocab)] # Match the shape of the vocabulary.

                # Create a sparse tensor where only the boosted tokens have non-zero values.
                boosted_score = tf.sparse.SparseTensor(ids_to_boost, values_tensor, shape)
                boosted_score = tf.sparse.reorder(boosted_score) # Organize the sparse tensor nicely.

                # Step 2: Transform the sparse tensor to a dense format, so it plays nice with other arrays.
                dense_vocab = tf.sparse.to_dense(boosted_score)

                # Step 3: Reshape it to [1, vocab_size] so it can merge with scores.
                resized_dense_vocab = tf.expand_dims(dense_vocab, axis = 0)

                # Step 4: Convert to a NumPy array for further tweaks.
                beam_keyword_boost = resized_dense_vocab.numpy()

                # Add the boosts for this topic to the beam. First topic? Start fresh. Otherwise, stack 'em up.
                if type(new_beam) == list:
                    new_beam = beam_keyword_boost
                else:
                    new_beam += beam_keyword_boost

            boosted_scores.append(new_beam) # Add the fully boosted beam to our collection

        # Squeeze out unnecessary dimensions to make it compatible with scores.
        boosted_scores = np.array(boosted_scores)
        boosted_scores = np.squeeze(boosted_scores, axis = (1,))

        # Step 5: Add the boosted scores to the original scores.
        # These scores just went from "meh" to "VIP lounge access."
        numpy_scores = scores.detach().numpy()  # Convert PyTorch tensor to NumPy array.
        new_scores = numpy_scores + boosted_scores # Apply the boosts.
        boosted_scores = torch.tensor(new_scores) # Transform back into a PyTorch tensor.

        return boosted_scores
