"""
Unit tests for WOPE poetry generation system.

Run with: python -m pytest test_wope.py -v
or: python test_wope.py
"""

import unittest
import numpy as np
import torch
from utils import get_similarity, ban_scores, boost_scores
from constants import EOS_TOKEN_ID, PAD_TOKEN_ID, DEFAULT_NUM_VERSES


class TestConstants(unittest.TestCase):
    """Test that constants are properly defined."""

    def test_token_ids(self):
        """Verify token ID constants match GPT-2 expectations."""
        self.assertEqual(EOS_TOKEN_ID, 50256)
        self.assertEqual(PAD_TOKEN_ID, 0)

    def test_defaults(self):
        """Verify default configuration values."""
        self.assertEqual(DEFAULT_NUM_VERSES, 4)
        self.assertIsInstance(DEFAULT_NUM_VERSES, int)


class TestSimilarity(unittest.TestCase):
    """Test semantic similarity functions."""

    def test_get_similarity_with_arrays(self):
        """Test cosine similarity calculation with numpy arrays."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        similarity = get_similarity(vec1, vec2, None, None)
        self.assertAlmostEqual(similarity, 1.0, places=5)

    def test_get_similarity_orthogonal(self):
        """Test similarity of orthogonal vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = get_similarity(vec1, vec2, None, None)
        self.assertAlmostEqual(similarity, 0.0, places=5)

    def test_get_similarity_opposite(self):
        """Test similarity of opposite vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        similarity = get_similarity(vec1, vec2, None, None)
        self.assertAlmostEqual(similarity, -1.0, places=5)


class TestBanScores(unittest.TestCase):
    """Test token banning functionality."""

    def test_ban_scores_basic(self):
        """Test basic token banning."""
        scores = torch.randn(2, 10)  # 2 beams, 10 tokens
        banned_tokens = [[1, 2], [3, 4]]  # Ban different tokens per beam

        result = ban_scores(scores, banned_tokens)

        # Banned tokens should be -inf
        self.assertEqual(result[0, 1], float('-inf'))
        self.assertEqual(result[0, 2], float('-inf'))
        self.assertEqual(result[1, 3], float('-inf'))
        self.assertEqual(result[1, 4], float('-inf'))

        # Non-banned tokens should be unchanged
        self.assertEqual(result[0, 0], scores[0, 0])
        self.assertEqual(result[1, 0], scores[1, 0])

    def test_ban_scores_empty(self):
        """Test banning with no tokens."""
        scores = torch.randn(2, 10)
        banned_tokens = [[], []]

        result = ban_scores(scores, banned_tokens)

        # Should return original scores unchanged
        torch.testing.assert_close(result, scores)

    def test_ban_scores_single_beam(self):
        """Test banning with single beam."""
        scores = torch.randn(1, 10)
        banned_tokens = [[5, 6, 7]]

        result = ban_scores(scores, banned_tokens)

        self.assertEqual(result[0, 5], float('-inf'))
        self.assertEqual(result[0, 6], float('-inf'))
        self.assertEqual(result[0, 7], float('-inf'))


class TestBoostScores(unittest.TestCase):
    """Test token boosting functionality."""

    def test_boost_scores_basic(self):
        """Test basic token boosting."""
        scores = torch.zeros(2, 10)  # 2 beams, 10 tokens
        vocab = {f'token_{i}': i for i in range(10)}

        cos_sim_dict = {
            'keyword1': {'intensity': 5, 'vocab': [1, 2, 3]},
            'keyword2': {'intensity': 10, 'vocab': [4, 5]}
        }

        tokens_to_boost = [
            [[1, 2, 3], [4, 5]],  # Beam 0: boost both keywords
            [[], [4, 5]]           # Beam 1: boost only keyword2
        ]

        result = boost_scores(scores, cos_sim_dict, tokens_to_boost, vocab)

        # Beam 0 should have boosts
        self.assertAlmostEqual(result[0, 1].item(), 5.0, places=3)
        self.assertAlmostEqual(result[0, 2].item(), 5.0, places=3)
        self.assertAlmostEqual(result[0, 4].item(), 10.0, places=3)

        # Beam 1 should only have keyword2 boosts
        self.assertAlmostEqual(result[1, 1].item(), 0.0, places=3)
        self.assertAlmostEqual(result[1, 4].item(), 10.0, places=3)

    def test_boost_scores_no_config(self):
        """Test boosting with no configuration."""
        scores = torch.randn(2, 10)
        vocab = {f'token_{i}': i for i in range(10)}

        result = boost_scores(scores, None, [], vocab)

        # Should return original scores unchanged
        torch.testing.assert_close(result, scores)

    def test_boost_scores_device_handling(self):
        """Test that boost scores properly handles device placement."""
        if torch.cuda.is_available():
            scores = torch.zeros(1, 10).cuda()
            vocab = {f'token_{i}': i for i in range(10)}
            cos_sim_dict = {'keyword': {'intensity': 5, 'vocab': [1, 2]}}
            tokens_to_boost = [[[1, 2]]]

            result = boost_scores(scores, cos_sim_dict, tokens_to_boost, vocab)

            # Result should be on same device as input
            self.assertEqual(result.device, scores.device)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""

    def test_ban_and_boost_together(self):
        """Test that banning and boosting work together correctly."""
        scores = torch.zeros(1, 10)
        vocab = {f'token_{i}': i for i in range(10)}

        # Boost some tokens
        cos_sim_dict = {'keyword': {'intensity': 5, 'vocab': [1, 2, 3]}}
        tokens_to_boost = [[[1, 2, 3]]]
        scores = boost_scores(scores, cos_sim_dict, tokens_to_boost, vocab)

        # Ban some tokens (including one that was boosted)
        banned_tokens = [[2, 4, 5]]
        scores = ban_scores(scores, banned_tokens)

        # Check results
        self.assertAlmostEqual(scores[0, 1].item(), 5.0, places=3)  # Boosted, not banned
        self.assertEqual(scores[0, 2].item(), float('-inf'))  # Boosted AND banned
        self.assertEqual(scores[0, 4].item(), float('-inf'))  # Only banned


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
