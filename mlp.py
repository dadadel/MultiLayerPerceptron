#!/usr/bin/env python3

import numpy as np
try:
    import matplotlib.pyplot as mp
except:
    mp = None


def sigmoid(x):
    return 1/(1+np.exp(-x))


def deriv_sigmoid(x):
    a = sigmoid(x)
    return a * (1 - a)


def tanh(x):
    ep = np.exp(x)
    en = np.exp(-x)
    #print("ep:{}\nen:{}\n".format(ep,en))
    return (ep - en)/(ep + en)


def deriv_tanh(x):
    a = tanh(x)
    return 1 - (a * a)


def relu(x):
    return np.maximum(x, np.zeros(x.shape))

def deriv_relu(x):
    ret = x
    ret[ret > 0] = 1
    ret[ret < 0] = 0
    return ret

def leaky_relu(x):
    ret = 0.01 * x
    #fixme should map to compare
    if x > 0:
        ret = x
    elif type(x) is np.ndarray:
        ret = np.ones(x.shape)*0.01
    return ret

def softmax(x):
    t = np.exp(x)
    sum_t = np.sum(t, axis=0)
    return t / sum_t

def w_rand_tanh(n, l, xavier_init=True):
    """ Initialize weights of a layer for tanH activation function

    :param n: vector of number of units per layer
    :param l: current layer
    :param xavier_init: if True will use Xavier initialization

    """
    if xavier_init:
        print("tanh factor={}".format(np.sqrt(1/n[l-1])))
        ret = np.random.randn(n[l], n[l-1]) * np.sqrt(1/n[l-1])
    else:
        print("tanh factor={}".format(np.sqrt(2/(n[l-1]+n[l]))))
        ret = np.random.randn(n[l], n[l-1]) * np.sqrt(2/(n[l-1]+n[l]))
    return ret

def w_rand_relu(n, l):
    """ Initialize weights of a layer for tanH activation function

    :param n: vector of number of units per layer
    :param l: current layer

    """
    print("relu factor={}".format(np.sqrt(2/n[l-1])))
    return np.random.randn(n[l], n[l-1]) * np.sqrt(2/n[l-1])

def w_rand_sigmoid(n, l):
    print("sigmoid factor={}".format(1/(n[l-1]*n[l])))
    return np.random.randn(n[l], n[l-1]) * (1/(n[l-1]*n[l]))

def w_rand_softmax(n, l, factor=0.01):
    print("softmax factor={}".format(factor))
    return np.random.randn(n[l], n[l-1]) * factor


class MultiLayerPerceptron(object):

    functions = {
        "sigmoid": {"function": sigmoid, "derivative": deriv_sigmoid, "w_rand": w_rand_sigmoid, "name": "sigmoid"},
        "tanh": {"function": tanh, "derivative": deriv_tanh, "w_rand": w_rand_tanh, "name": "tanh"},
        "relu": {"function": relu, "derivative": deriv_relu, "w_rand": w_rand_relu, "name": "relu"},
        "softmax": {"function": softmax, "derivative": None, "w_rand": w_rand_softmax, "name": "softmax"},
    }

    def __init__(self, L=1, n=None, g=None, alpha=0.01, set_random_w=True, use_formula_w=False, w_rand_factor=1):
        """Initializes network geometry and parameters
        :param L: number of layers including output and excluding input. Defaut 1.
        :type L: int
        :param n: list of number of units per layer including input. Default [2, 1].
        :type n: list
        :param g: list of activation functions name per layer excluding input.
            Possible names are: "sigmoid", "tanh". Default ["sigmoid"].
        :type g: list
        :param alpha: learning rate. Default 0.01.
        :param set_random_w: if True will initialize randomly weights using either w_rand_factor
            or a formula depending on activation functions if use_formula_w is True
        """
        #w_rand_factor = 1
        self._prepared = False
        self._L = L
        if n is None:
            n = [2, 1]
        self._n = n
        if g is None:
            g = [MultiLayerPerceptron.functions["sigmoid"]]
        else:
            g = [MultiLayerPerceptron.functions[fct] for fct in g]
        self._g = [None] + g
        # check if softmax multi-class classification
        self._softmax = False
        if g[-1]["name"] == "softmax":
            self._softmax = True
        self._A = None
        self._X = None
        self._Y = None
        self._Z = None
        self._m = 0
        self._alpha = alpha
        # optimization
        self._lambda = 0
        self._regularization = False
        self._momentum = False
        self._rmsprop = False
        self._adam = False
        # initialise weights
        self._b = [None] + [np.zeros((n[l+1], 1)) for l in range(L)]
        self._W = [None] + [np.zeros((n[l+1], n[l])) for l in range(L)]
        if set_random_w:
            self.init_random_weights(use_formula_w, w_rand_factor)
        assert(len(self._g) == len(self._W))
        assert(len(self._g) == len(self._b))
        assert(len(self._g) == len(self._n))

    def init_random_weights(self, use_formula=False, w_rand_factor=1):
        """Initialize randomly weights using a factor or using some formula
        
        :param w_rand_factor: factorize random weights with this (default 1)
        :param use_formula: if True will use formules corresponding to the activation functions
        """
        if use_formula:
            for l0 in range(self._L):
                l = l0 + 1
                self._W[l] = self._g[l]["w_rand"](self._n, l)
        else:
            if type(w_rand_factor) is list:
                self._W = [None] + [np.random.randn(self._n[l+1], self._n[l])*w_rand_factor[l] for l in range(self._L)]
            else:
                self._W = [None] + [np.random.randn(self._n[l+1], self._n[l])*w_rand_factor for l in range(self._L)]


    def use_regularization(self, lambd):
        """Activates regularization for backpropagation

        :param lambd: the lambda parameter value for regularization

        """
        self._regularization = True
        self._lambda_regul = lambd

    def use_momentum(self, beta=0.9, v_dw=0., v_db=0.):
        """Activates momentum optimization for backpropagation

        :param beta: the beta parameter value for momentum (default 0.9)
        :param v_dw: v_dw initial value for momentum (default 0.0)
        :param v_db: v_db initial value for momentum (default 0.0)

        """
        self._momentum = True
        self._beta_momentum = beta
        n = self._n
        self._v_dw_momentum = [None] + [v_dw * np.ones((n[l+1], n[l])) for l in range(self._L)]
        self._v_db_momentum = [None] + [v_db * np.ones((n[l+1], 1)) for l in range(self._L)]

    def use_rmsprop(self, beta=0.999, s_dw=0., s_db=0., epsilon=1.0e-8):
        """Activates RMSProp optimization for backpropagation

        :param beta: the beta parameter value for RMSProp (default 0.999)
        :param s_dw: s_dw initial value for RMSProp (default 0.0)
        :param s_db: s_db initial value for RMSProp (default 0.0)
        :param epsilon: epsilon value for RMSProp (default 1.0e-8)

        """
        self._rmsprop = True
        self._beta_rmsprop = beta
        n = self._n
        self._s_dw_rmsprop = [None] + [s_dw * np.ones((n[l+1], n[l])) for l in range(self._L)]
        self._s_db_rmsprop = [None] + [s_db * np.ones((n[l+1], 1)) for l in range(self._L)]
        self._epsilon_rmsprop = epsilon

    def use_adam(self, beta_m=0.9, v_dw=0., v_db=0., beta_r=0.999, s_dw=0., s_db=0., epsilon=1.0e-8):
        """Activates Adam optimization for backpropagation

        :param beta_m: the beta parameter value for momentum (default 0.9)
        :param v_dw: v_dw initial value for momentum (default 0.0)
        :param v_db: v_db initial value for momentum (default 0.0)
        :param beta_r: the beta parameter value for RMSProp (default 0.999)
        :param s_dw: s_dw initial value for RMSProp (default 0.0)
        :param s_db: s_db initial value for RMSProp (default 0.0)
        :param epsilon: epsilon value for RMSProp (default 1.0e-8)

        """
        self._adam = True
        self.use_momentum(beta_m, v_dw, v_db)
        self.use_rmsprop(beta_r, s_dw, s_db, epsilon)

    def set_all_input_examples(self, X, m=None):
        """Set the input examples.

        :param X: matrix of dimensions (n[0], m). Accepts also a list (len m) of lists (len n[0])
        :param m: number of training examples.
        :type m: int
        """
        if m is None:
            m = self._m
        if type(X) is list:
            assert(len(X) == m)
            self._X = np.matrix(X).T
        else:
            #print(X.shape, self._n[0], m)
            assert(X.shape == (self._n[0], m))
            self._X = X
        self._m = m
        assert((self._m == m) or (self._m == 0))
        self._m = m
        self._prepared = False

    def set_all_expected_output_examples(self, Y, m=None):
        """Set the output examples

        :param Y: matrix of dimensions (n[L], m). Accepts also a list (len m) of lists (len n[L])
        :param m: number of training examples.
        :type m: int
        """
        if m is None:
            m = self._m
        if type(Y) is list:
            assert(len(Y) == m)
            self._Y = np.matrix(Y).T
        else:
            #print(Y.shape, self._n[self._L], m)
            assert(Y.shape == (self._n[self._L], m))
            self._Y = Y
        assert((self._m == m) or (self._m == 0))
        self._m = m

    def set_all_training_examples(self, X, Y, m=1):
        """Set all training examples

        :param X: matrix of dimensions (n[0], m). Accepts also a list (len m) of lists (len n[0])
        :param Y: matrix of dimensions (n[L], m). Accepts also a list (len m) of lists (len n[L])
        :param m: number of training examples.
        :type m: int
        """
        self._m = m
        self.set_all_input_examples(X, m)
        self.set_all_expected_output_examples(Y, m)

    def prepare(self, X=None, m=None, force=False):
        """Prepare network for propagation"""
        if X is not None:
            force = True
        #print("sforce,prep =", force, self._prepared)
        if force or self._prepared == False:
            if not force:
                self._prepared = True
            if X is None:
                assert(self._X is not None)
                X = self._X
                if m is None:
                    m = self._m
                if m == 0:
                    if type(X) is list:
                        m = 1
                    else:
                        m = X.shape[1]
            else:
                if m is None:
                    if type(X) is list:
                        m = 1
                    else:
                        m = X.shape[1]
                if type(X) is list:
                    X = np.array(X).reshape(len(X), 1)
            if m is None:
                assert(self._m > 0)
                m = self._m
            #print("m prep =", m)
            self._A = [X]
            self._A += [np.empty((self._n[l+1], m)) for l in range(self._L)]
            self._Z = [None] + [np.empty((self._n[l+1], m)) for l in range(self._L)]

    def propagate(self):
        """Forward propagation

        :return: matrix of computed outputs (n[L], m)

        """
        for l0 in range(self._L):
            l = l0 + 1
            self._Z[l] = np.dot(self._W[l], self._A[l-1]) + self._b[l]
            self._A[l] = self._g[l]["function"](self._Z[l])
        return self._A[self._L]

    def compute_outputs(self, X=None, m=None):
        """Compute outputs with forward propagation.
        Note: if no input provided, then the input should have been set using
        either `set_all_input_examples()` or `set_all_training_examples()`.

        :param X: if None will use self._X
        :return: the computed output

        """
        if X is not None:
            if m is None:
                if type(X) is list:
                    m = 1
                else:
                    m = X.shape[1]
            print("lenX,m",len(X),m)
            self.prepare(X, m)
        else:
            self.prepare()
        self.propagate()
        return self._A[self._L]

    def get_output(self):
        return self._A[self._L]

    def get_expected_output(self):
        return self._Y

    def get_input(self):
        return self._X

    def get_weights(self):
        return self._W[1:]

    def get_bias(self):
        return self._b[1:]

    def set_flatten_weights(self, W):
        """Set weights from a flatten list"""
        shapes = [w.shape for w in self._W[1:]]
        sizes = [w.size for w in self._W[1:]]
        flat_size = sum(sizes)
        assert(len(W) == flat_size)
        ini = 0
        for i, (size, shape) in enumerate(zip(sizes, shapes)):
            self._W[1+i] = np.reshape(W[ini:ini+size], shape)
            ini += size

    def set_flatten_bias(self, B):
        """Set bias from a flatten list"""
        shapes = [b.shape for b in self._b[1:]]
        sizes = [b.size for b in self._b[1:]]
        flat_size = sum(sizes)
        assert(len(B) == flat_size)
        ini = 0
        for i, (size, shape) in enumerate(zip(sizes, shapes)):
            self._b[1+i] = np.reshape(B[ini:ini+size], shape)
            ini += size

    def back_propagation(self, get_cost_function=False):
        """Back propagation

        :param get_cost_function: if True the cost function J
            will be computed and returned.
            J = -1/m((Y(A.T)) + (1-Y)(A.T))
            if self._regularization will add:
            J += lamda/(2*m)*Wnorm
        :return: the cost function if get_cost_function==True else None

        """
        J = None
        L = self._L
        m = self._m
        dW = [None] + [None] * self._L
        db = [None] + [None] * self._L
        dA = [None] + [None] * self._L
        dA[L] = -self._Y/self._A[L] + ((1-self._Y)/(1-self._A[L]))

        # Compute cost function
        if get_cost_function:
            if self._softmax:
                # case of softmax multi-class
                loss = -np.sum(self._Y * np.log(self._A[L]), axis=0)
                J = 1/m * np.sum(loss)
            else:
                J = -1/m * np.sum(( np.dot(self._Y, np.log(self._A[L]).T) + \
                             np.dot((1 - self._Y), np.log(1-self._A[L]).T) ), axis=1)
            # add regularization
            if self._regularization:
                wnorms = 0
                for w in self._W[1:]:
                    wnorms += np.linalg.norm(w)
                J += self._lambda_regul/(2*m) * wnorms

        # Compute weights derivatives
        for l in range(L, 0, -1):
            if self._softmax and l == L:
                # output layer for softmax multi-class
                dZ = self._A[L] - self._Y
            else:
                dZ = dA[l] * self._g[l]["derivative"](self._Z[l])
            dW[l] = 1/self._m * np.dot(dZ, self._A[l-1].T)
            db[l] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            dA[l-1] = np.dot(self._W[l].T, dZ)

        # Update weights
        for l in range(L, 0, -1):
            w_factor = dW[l]
            b_factor = db[l]
            # add momentum
            if self._momentum:
                self._v_dw_momentum[l] = self._beta_momentum * self._v_dw_momentum[l] + \
                                      (1 - self._beta_momentum) * dW[l]
                self._v_db_momentum[l] = self._beta_momentum * self._v_db_momentum[l] + \
                                      (1 - self._beta_momentum) * db[l]
                w_factor = self._v_dw_momentum[l]
                b_factor = self._v_db_momentum[l]
            # add RMSProp
            if self._rmsprop:
                self._s_dw_rmsprop[l] = self._beta_rmsprop * self._s_dw_rmsprop[l] + \
                                      (1 - self._beta_rmsprop) * (dW[l]**2)
                self._s_db_rmsprop[l] = self._beta_rmsprop * self._s_db_rmsprop[l] + \
                                      (1 - self._beta_rmsprop) * (db[l]**2)
                # if adam optimization is use the formula will work as w/b_factor are set in momentum
                w_factor = w_factor / (np.sqrt(self._s_dw_rmsprop[l]) + self._epsilon_rmsprop)
                b_factor = b_factor / (np.sqrt(self._s_db_rmsprop[l]) + self._epsilon_rmsprop)
            # add regularization
            if self._regularization:
                self._W[l] = self._W[l] - self._alpha * w_factor - \
                             (self._alpha*self._lambda_regul/m) * self._W[l]
            else:
                self._W[l] = self._W[l] - self._alpha * w_factor
            self._b[l] = self._b[l] - self._alpha * b_factor

        return J

    def minimize_cost(self, min_cost, max_iter=100000, alpha=None, plot=False):
        """Propagate forward then backward in loop while minimizing the cost function.

        :param min_cost: cost function value to reach in order to stop algo.
        :param max_iter: maximum number of iterations to reach min cost befor stoping algo. (Default 100000).
        :param alpha: learning rate, if None use the instance alpha value. Default None.

        """
        nb_iter = 0
        if alpha is None:
            alpha = self._alpha
        self.propagate()
        if plot:
            y=[]
            x=[]
        for i in range(max_iter):
            J = self.back_propagation(True)
            if plot:
                y.append(J)
                x.append(nb_iter)
            self.propagate()
            nb_iter = i + 1
            if J <= min_cost:
                break
        if mp and plot:
            mp.plot(x,y)
            mp.show()
        return {"iterations": nb_iter, "cost_function": J}

    def learning(self, X, Y, m, min_cost=0.05, max_iter=100000, alpha=None, plot=False):
        """Tune parameters in order to learn examples by propagate and backpropagate.

        :param X: the inputs training examples
        :param Y: the expected outputs training examples
        :param m: the number of examples
        :param min_cost: cost function value to reach in order to stop algo. Default 0.0.5
        :param max_iter: maximum number of iterations to reach min cost befor stoping algo. (Default 100000).
        :param alpha: learning rate, if None use the instance alpha value. Default None.

        """
        self.set_all_training_examples(X, Y, m)
        self.prepare()
        res = self.minimize_cost(min_cost, max_iter, alpha, plot)
        return res

