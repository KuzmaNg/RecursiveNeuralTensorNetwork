# rntn.py
"""
    rntn
    ~~~~~
    A deep learning recursive neural tensor network
    for sentiment analysis, implemented from the paper:

    "Recursive Deep Models for Semantic Compositionality
     Over a Sentiment Treebank" by Socher et al.

    Variables are defined as in the paper's notation:
    L, Ws, W, V, etc.

    tanh is used as the activation function.
"""
import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import softmax
from data_transformation import DataTransformation, DEV_TXT, TRAIN_TXT




class RecursiveNeuralTensorNetwork(object):
    

    def __init__(self, dt, rng,
                 d=25, C=5, V=None, W=None, b=None, L=None, Ws=None,
                 activation=T.tanh):
        self.d = d
        self.C = C
        self.vsize = len(dt.vocab_list)
        print 'The provided vocabulary has %i words' % self.vsize
        self.dt = dt
        n_in = 10 # use this as an order of magnitude for now
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(1.0 / n_in),
                    high=numpy.sqrt(1.0 / n_in),
                    size=(d, 2*d)),
                dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W',
                              borrow=True)
        # possible bias
        if b is None:
            b_values=numpy.zeros((d,), 
                                 dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b',
                              borrow=True)
        r = 0.0001
        if L is None:
            L_values = numpy.asarray(
                rng.uniform(
                    low=-r,
                    high=r,
                    size=(d, self.vsize)),
                dtype=theano.config.floatX)
            L = theano.shared(value=L_values,
                              name='L', borrow=True)
        if Ws is None:
            Ws_values = numpy.asarray(
                rng.uniform(
                    low=-r,
                    high=r,
                    size=(C, d)),
                dtype=theano.config.floatX)
            Ws = theano.shared(value=Ws_values,
                               name='Ws', borrow=True)
        # rearrange indices so V in R^(d*2d*2d)
        if V is None:
            V_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(1.0 / (n_in)**2),
                    high=numpy.sqrt(1.0 / (n_in)**2),
                    size=(d, 2*d, 2*d)),
                dtype=theano.config.floatX)
            V = theano.shared(value=V_values,
                              name='V', borrow=True)        
        self.W = W
        self.b = b
        self.L = L
        self.Ws = Ws
        self.V = V
        # list of parameters
        self.theta = [self.W, self.b, self.L, self.Ws, self.V]
        self.lambda_const = 0.01 
        self.param_error = self.lambda_const * ((self.V**2).sum() + \
                                           (self.W**2).sum() + \
                                           (self.Ws**2).sum() + \
                                           (self.L**2).sum() + \
                                           (self.b**2).sum())
        ym = T.dmatrix('ym')
        tm = T.dmatrix('tm')
        self.cross_entropy = theano.function([ym, tm],
                                             (tm * T.log(ym)).sum())
        v1 = T.dvector('v1')
        v2 = T.dvector('v2')
        # direct sum
        v = T.concatenate([v1,v2])
        pair_output = T.dot(T.dot(self.V, v), v) + T.dot(self.W,
                                                         v) + b
        self.pair_map = theano.function([v1,v2], pair_output)
        self.softmax = theano.function([v1], softmax(T.dot(Ws, v1)).flatten())
        

    def forward_pass(self, inpt):
        """Forward passes for the given list of
        word indices."""
        inpt_tree = self.dt.get_tree(inpt)
        actuals = self.dt.tree_scores(inpt_tree)
        zero_one_actuals = [numpy.zeros((self.C,)) for e in actuals]
        for i, e in enumerate(zero_one_actuals):
            e[actuals[i]] = 1
        stack = self.dt.tree_stack(inpt_tree)
        outputs = [numpy.zeros(shape=(self.d,))] * len(stack)
        i = len(stack) - 1
        while len(stack) > 1:
            nr = stack.pop()
            nl = stack.pop()
            nr_in = self.dt.get_word_indices(nr)
            nl_in = self.dt.get_word_indices(nl)
            if len(nr_in) == 1:
                vr = self.L.get_value()[:, nr_in].flatten()
                outputs[i] = vr
            else:
                vr = outputs[i]
            if len(nl_in) == 1:
                vl = self.L.get_value()[:, nl_in].flatten()
                outputs[i-1] = vl
            else:
                vl = outputs[i-1]
            p = self.pair_map(vl, vr)
            i -= 2
            possible_parent = stack[i]
            j = 0
            while nr not in possible_parent.subtrees:
                j += 1
                possible_parent = stack[i-j]
            parent = possible_parent
            outputs[i-j] = p
        assert [e.shape == (self.d,) for e in outputs]
        # apply softmax
        outputs = [self.softmax(e) for e in outputs]        
        assert [e.shape == (self.C,) for e in outputs]
        assert (len(outputs) == len(zero_one_actuals))
        actuals_matrix = numpy.asarray(zero_one_actuals).transpose()
        outputs_matrix = numpy.asarray(outputs).transpose()
        self.error = self.cross_entropy(outputs_matrix, actuals_matrix) + \
                     self.param_error
        return self.error


def main():
    pass


if __name__ == '__main__':
    main()

    
    
