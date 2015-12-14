# test_data_transformation.py

"""
    test_data_transformation
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Unit tests for the DataTransformation class

"""
from rntn import data_transformation

import random
import unittest


class TestDataTransformation(unittest.TestCase):

    def setUp(self):
        """ Creates a DataTransformation instance."""
        self.dt = data_transformation.DataTransformation()

    def tearDown(self):
        pass

    def test_tree_iterator(self):
        pass

    def test_get_vocab_list(self):
        """ Verify that all words have been included. """
        vocab_list = self.dt.get_vocab_list()
        assert (len(vocab_list) == 5374)
        
    def test_get_word_index(self):
        pass

    def test_tree_dict(self):
        pass

    def test_tree_scores(self):
        pass

    def test_tree_stack(self):
        pass

    def test_get_word_indices(self):
        """ Check that the words corresponding to the given
        returned word indices are indeed in the provided tree."""
        n = len(self.dt.all_ptb_trees)
        random_tree = self.dt.all_ptb_trees[random.randint(0, n)]
        word_indices = self.dt.get_word_indices(random_tree)
        for i in word_indices:
            word = self.dt.vocab_list[i]
            assert (word in str(random_tree))
        

    def test_get_tree(self):
        """Checks consistency with .get_word_indices() """
        n = len(self.dt.all_ptb_trees)
        random_tree = self.dt.all_ptb_trees[random.randint(0, n)]
        word_indices = self.dt.get_word_indices(random_tree)
        random_tree2 = self.dt.get_tree(word_indices)
        assert (str(random_tree) == str(random_tree2))




if __name__ == "__main__":
    unittest.main()
    
