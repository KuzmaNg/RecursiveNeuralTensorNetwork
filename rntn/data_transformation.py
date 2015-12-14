# data_transformation.py
"""

    data_transformation
    ~~~~~~~~~~~~~~~~~~~
    A class for transforming PTB files to give
    to the neural network in the 
    RecursiveNeuralTensorNetwork class.
    
"""

from ptb import *

import os


MODULE_PATH = os.path.dirname(__file__)
DEV_TXT = MODULE_PATH + '/trees/dev.txt'
TRAIN_TXT = MODULE_PATH + '/trees/train.txt'


class DataTransformation(object):
    
    """Class for transforming PTB data
    for use in our neural network."""

    def __init__(self, ptb_file=DEV_TXT):
        """Creates and stores in a list tree objects
        for all sentences in the given PTB file, 
        collects the vocabulary list for the given PTB
        file, and creates a dictionary mapping
        vocabulary word indices for the given
        sentence to a tree object."""
        self.all_ptb_trees = []
        self.vocab_list = []
        self.word_indices_hash = {}
        with open(ptb_file, 'r') as ptb:
            all_trees = ptb.read()
            nl = 0
            nr = 0
            sentence = ''
            for e in all_trees:
                if e == '(':
                    nl += 1
                    sentence += e
                    continue
                if e == ')':
                    nr += 1
                    sentence += e
                    if nl == nr:
                        s = sentence.strip()
                        tree = PTB_Tree()
                        tree.set_by_text(s)
                        self.all_ptb_trees.append(tree)
                        sentence = ''
                    continue
                else:
                    sentence += e
        vocab_set = set()
        for t in self.all_ptb_trees:
            s = t.word_yield()
            for e in s.split():
                vocab_set.add(e)
        self.vocab_list = sorted(list(vocab_set))

        for t in self.all_ptb_trees:
            word_indices = self.get_word_indices(t)
            self.word_indices_hash[tuple(word_indices)] = t
            
    
    def tree_iterator(self):
        """Returns an iterator over all trees
        in the file."""
        for t in self.all_ptb_trees:
            yield t

    def get_vocab_list(self):
        """Returns the list of unique words in the file."""
        return self.vocab_list
    

    def get_word_index(self, word):
        """Returns the index of the given word in 
        the vocabulary."""
        try:
            return self.vocab_list.index(word)
        except:
            return -1

    def get_tree_dict(self, tree):
        """Returns a dictionary of 
        <tree word indices (list), classification(int)>
        for the given tree."""
        t_dict = {}
        def traverse_tree(t):
            cls = t.__repr__()[1]
            try:
                float(cls)
            except:
                cls = 0
            t_dict[self.get_word_indices(t)] = int(cls)
            if len(t.subtrees) > 0:
                for e in t.subtrees:
                    traverse_tree(e)
        traverse_tree(tree)
        return t_dict

    def tree_scores(self, tree):
        """Returns the classifications
        for the given tree as a list to
        be treated as a stack containing 
        the classification for each node,
        with classification for leaves on
        the top of the stack."""
        visited, queue = set(), [tree]
        stack = []
        while queue:
            t = queue.pop()
            if t not in visited:
                visited.add(t)
                cls = t.__repr__()[1]
                try:
                    float(cls)
                except:
                    cls = 0
                stack = stack + [int(cls)]
                for ct in t.subtrees:
                    if ct not in visited:
                        queue = [ct] + queue
        return stack

    def tree_stack(self, tree):
        """For forward passing. Returns
        a list to be treated as a stack containing
        all nodes (as subtrees) for the tree, ordered
        with the leaves on the top of the stack."""
        visited, queue = set(), [tree]
        stack = []
        while queue:
            t = queue.pop()
            if t not in visited:
                visited.add(t)
                stack = stack + [t]
                for ct in t.subtrees:
                    if ct not in visited:
                        queue = [ct] + queue
        return stack

    def get_word_indices(self, tree):
        """Returns a tuple of the indices of the words
        in the given tree."""
        sentence = tree.word_yield()
        return tuple([self.get_word_index(w) for w in 
                sentence.split()])
    
    def get_tree(self, word_indices):
        """Returns the tree corresponding to the given
        indices."""
        return self.word_indices_hash[tuple(word_indices)]
        
    

def main():
    dt = DataTransformation()
    t = dt.all_ptb_trees[0]
    s = dt.tree_stack(t)
    while len(s) > 0:
        print s.pop()





if __name__ == '__main__':
    main()


            
