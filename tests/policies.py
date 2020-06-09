import numpy as np
import itertools as it
from scipy.special import softmax

class FourierPolicy:
    def __init__(self, deg, n_actions, state_dim):
        self.deg = deg
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.coeffs = [ # http://irl.cs.brown.edu/fb.php
            list(reversed(e)) for e
            in list(it.product(range(0, self.deg + 1), repeat=self.state_dim))
        ]
        self.params = np.random.rand(n_actions, (deg + 1)**state_dim)

    def get_params(self):
        """ Returns the full parameter array """
        return self.params

    def get_action_params(self, a):
        """ Returns the parameter vector associated with the action `a` """
        return self.params[a]

    def with_params(self, theta):
        """ Sets the parameter array and returns self for ease of chaining """
        self.params = theta
        return self

    def basify(self, s):
        """ Maps a normalized state s to a vector of features \phi(s) """
        return np.array([np.cos(np.pi * np.dot(c, s)) for c in self.coeffs])

    def eval(self, s):
        """ Returns the probabilities under this policy of each action """
        phi_s = self.basify(s)
        probs = softmax(np.dot(self.params, phi_s))
        return probs

    def eval_one(self, s, a):
        """ Returns the probability under this policy to take action `a` from state `s` """
        return self.eval(s)[a]

    def sample(self, s):
        """ Return an action sampled from the policy conditioned on state `s` """
        rng = np.random.default_rng()
        return rng.choice(np.arange(self.n_actions), p=self.eval(s))

