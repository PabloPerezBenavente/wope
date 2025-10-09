import pronouncing
import numpy as np
import torch
from numpy.linalg import norm
from constants import TOP_SIMILAR_TOKENS

def get_embedding(word, vocab, model):
    """
    Retrieves the embedding vector for a given word from the model's embedding layer.

    Args:
        word (str): The word to encode. Must exist in the vocabulary.
        vocab (dict): Dictionary mapping words to their token IDs.
        model (GPT2LMHeadModel): The language model containing pre-trained embeddings.

    Returns:
        torch.Tensor: The embedding vector for the specified word.
    """
    # Look up the word's token ID in the vocabulary
    token_id = vocab[word]

    # Retrieve the embedding from the model's word token embedding layer
    embedding = model.transformer.wte.weight[token_id]

    return embedding


def get_similarity(key, query, vocab, model):
    """
    Calculates the cosine similarity between two words or embedding vectors.

    Args:
        key (str or np.ndarray): The first word (string) or its embedding vector (numpy array).
        query (str or np.ndarray): The second word (string) or its embedding vector (numpy array).
        vocab (dict): Dictionary mapping words to token IDs.
        model (GPT2LMHeadModel): The language model containing pre-trained embeddings.

    Returns:
        float: Cosine similarity value between -1 and 1, where 1 indicates identical vectors.
    """
    # Convert word strings to embedding vectors if necessary
    if isinstance(key, str):
        key = get_embedding(key, vocab, model).detach().cpu().numpy()
    if isinstance(query, str):
        query = get_embedding(query, vocab, model).detach().cpu().numpy()

    # Calculate cosine similarity: dot product divided by product of magnitudes
    cosine_similarity = np.dot(query, key) / (norm(query) * norm(key))

    return cosine_similarity


def create_boosted_vocab(ids, value, vocab):
    """
    Creates a numpy array with boosted scores for specified token IDs.

    Args:
        ids (list): Token IDs to receive the boost.
        value (int): The boost value to add to the selected tokens' scores.
        vocab (dict): The complete vocabulary dictionary.

    Returns:
        np.ndarray: Array where specified tokens have the boost value,
            all other positions are zero.
    """
    # Create zero array matching vocabulary size
    boosted_vocab = np.zeros(len(vocab), dtype=np.float32)

    # Set boost values at specified indices
    for token_id in ids:
        boosted_vocab[token_id] = float(value)

    return boosted_vocab

def get_semantic_items(word, vocab, model):
    """
    Identifies the top 50 most semantically similar tokens to a given word.

    Only tokens with valid pronunciations (as determined by CMU Pronouncing Dictionary)
    are considered, ensuring the output is suitable for poetic composition.

    Args:
        word (str): The reference word to find semantic similarities for.
        vocab (dict): The complete vocabulary dictionary.
        model (GPT2LMHeadModel): The language model used for similarity calculations.

    Returns:
        list: Token IDs of the top 50 most semantically similar words.
    """
    similar_words = []
    for token in vocab:
        # Check if the token has a valid pronunciation
        if pronouncing.phones_for_word(token.strip(' Ġ').lower()):
            # Exclude the token if it's identical to or contains the input word
            if token.strip(' Ġ').lower() not in word and word not in token.strip(' Ġ').lower():
                similarity = get_similarity(token, word, vocab, model)
                similar_words.append((token, similarity))

    # Sort tokens by similarity score in descending order
    ordered_words = sorted(similar_words, key=lambda x: x[1], reverse=True)

    # Extract the top N most similar token IDs
    similar_tokens = [vocab[token[0]] for token in ordered_words[:TOP_SIMILAR_TOKENS]]

    return similar_tokens

def ban_scores(scores, banned_tokens):
    """
    Masks specified tokens by setting their logit scores to negative infinity.

    Args:
        scores (torch.Tensor): Logits distribution of shape (batch_size, vocab_size).
        banned_tokens (list): List of lists containing token IDs to ban for each batch.
            Shape: (batch_size, num_banned_tokens).

    Returns:
        torch.Tensor: Modified scores tensor with banned tokens set to -inf.
    """
    # Collect coordinates of all tokens to ban
    banned_mask_list = []

    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            # Store (batch_index, token_index) pairs
            banned_mask_list.append([idx, token])

    # If no tokens to ban, return original scores
    if not banned_mask_list:
        return scores

    # Convert list to tensor of banned coordinates
    banned_mask = torch.LongTensor(banned_mask_list)

    # Create indices tensor for sparse construction
    indices = torch.ones(len(banned_mask))

    # Build sparse tensor representing banned positions, then convert to dense boolean mask
    banned_mask = (
        torch.sparse.LongTensor(banned_mask.t(), indices, scores.size())
        .to(scores.device)
        .to_dense()
        .bool()
    )

    # Mask out banned tokens by setting their scores to negative infinity
    scores = scores.masked_fill(banned_mask, -float("inf"))

    return scores

def boost_scores(scores, cos_sim_dict, tokens_to_boost, vocab):
    """
    Increases logit scores for semantically similar tokens based on configuration.

    Args:
        scores (torch.Tensor): Logits distribution of shape (batch_size, vocab_size).
        cos_sim_dict (dict): Configuration for each keyword with structure:
            {'keyword': {'intensity': int, 'period': int, 'vocab': list}}
        tokens_to_boost (list): Nested list structure: [[[beam1_kw1], [beam1_kw2]], ...]
            indicating which tokens to boost for each beam and keyword.
        vocab (dict): The complete vocabulary dictionary.

    Returns:
        torch.Tensor: Modified scores with boosted values for specified tokens.
    """
    # Only apply boosts if configuration is provided
    if not cos_sim_dict:
        return scores

    keyword_list = list(cos_sim_dict.keys())
    batch_size = scores.shape[0]
    vocab_size = len(vocab)

    # Create boost tensor initialized to zeros
    boost_values = np.zeros((batch_size, vocab_size), dtype=np.float32)

    # Apply boosts for each beam
    for beam_idx, beam in enumerate(tokens_to_boost):
        for keyword_idx, topic_tokens in enumerate(beam):
            if topic_tokens:  # Only process if there are tokens to boost
                topic = keyword_list[keyword_idx]
                intensity = cos_sim_dict[topic]['intensity']

                # Add boost values for this keyword's tokens
                for token_id in topic_tokens:
                    boost_values[beam_idx, token_id] += intensity

    # Convert to PyTorch tensor and add to original scores
    boost_tensor = torch.from_numpy(boost_values).to(scores.device)
    boosted_scores = scores + boost_tensor

    return boosted_scores
