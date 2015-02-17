# learning.py
"""
    learning
    ~~~~~~~~
"""
from data_transformation import DataTransformation
import numpy
import pickle
from rntn import RecursiveNeuralTensorNetwork
import theano
import theano.tensor as T

SMALL_SET = 'trees/small_set.txt'
TRAIN_TXT = 'trees/train.txt'
DEV_TXT = 'trees/dev.txt'
TEST_TXT = 'trees/test.txt'
PARAMS_PICKLED = 'pickled_data/parameters.pkl'


class Learning(object):

    
    def __init__(self, training_file=TRAIN_TXT):
        """Loads training data and creates the recursive
        neural network."""
        self.dt = DataTransformation(training_file)
        self.num_training_samples = len(list(self.dt.tree_iterator()))
        self.rng = numpy.random.RandomState(1234)
        self.rntn = RecursiveNeuralTensorNetwork(self.dt,
                                                 self.rng)
    

    def batch_learn(self, training_iterations=100,
                    learning_rate=0.01):
        """Trains the neural network with batch learning."""
        print 'Training the neural network with batch learning...'
        self.mini_batch_learn(training_iterations=training_iterations,
                              mini_batch_size=self.num_training_samples,
                              learning_rate=learning_rate)
    

    def online_learn(self, training_iterations=100,
                     learning_rate=0.01):
        """Trains the neural network with online learning,
        using a mini-batch size of 1, that is, updating
        the weights after each input forward pass."""
        print 'Training the neural network with online learning...'
        self.mini_batch_learn(training_iterations=training_iterations,
                                 mini_batch_size=1,
                                 learning_rate=learning_rate)

    
    def mini_batch_learn(self, training_iterations=100,
                    mini_batch_size=10, learning_rate=0.01):
        """Trains the neural network using mini-batch learning."""
        for i in range(training_iterations):
            cost = 0.0
            tree_iterator = self.dt.tree_iterator()
            batch_num = 1
            for i, tree in enumerate(tree_iterator):
                indices = self.dt.get_word_indices(tree)
                cost += self.rntn.forward_pass(indices)
                if (i+1) % mini_batch_size == 0:
                    gparams = [T.grad(cost, theta) for theta
                               in self.rntn.theta]
                    updates = [(param, param - learning_rate * \
                                gtheta) for param, gtheta in 
                               zip(self.rntn.theta, gparams)]
                    for e in updates:
                        tmp_new = e[1].eval({})
                        e[0].set_value(tmp_new)
                    print 'Batch %i cost: %f' % (batch_num,
                                                 cost.eval({}))
                    batch_num += 1
                    cost = 0.0
        self._pickle_parameters()


    def _pickle_parameters(self):
        """Pickles the current values of the neural
        network parameters."""
        params = [e.get_value() for e in self.rntn.theta]
        with open(PARAMS_PICKLED, 'w') as f:
            pickle.dump(params, f)



    def _load_parameters(self):
        """Sets the neural network parameters
        to those values stored in the pickled file."""
        with open(PARAMS_PICKLED, 'r') as f:
            params = pickle.load(f)
        for i, e in enumerate(params):
            self.rntn.theta[i].set_value(e)
                                


    def test_learning(self, test_file=TEST_TXT):
        """Tests the neural network against a test file.
        Outputs the error value for each test input."""
        print 'Testing the neural network...'
        self._load_parameters()
        self.test_dt = DataTransformation(test_file)
        tree_iterator = self.test_dt.tree_iterator()
        cost = 0.0
        for tree in tree_iterator:
            indices = self.test_dt.get_word_indices(tree)
            cost += self.rntn.forward_pass(indices)
            print cost.eval({})

                
    
def main():
    ln = Learning(training_file=DEV_TXT)
    ln.online_learn(training_iterations=50, 
                    learning_rate=0.5)
    ln.test_learning(test_file=DEV_TXT)



if __name__ == '__main__':
    main()
