# pypi
from numba import jit
import numpy

@jit
def find_first(item, vector):
    """find the first item in the vector

    Args:
     item: thing to match
     vector: thing to search

    Returns:
     value: index of first matching item, -1 if not found
    """
    for index in range(len(vector)):
        if item == vector[index]:
            return index
    return -1
        

class EpsilonGreedy(object):
    """The Epsilon Greedy Algorithm

    Args:
     epsilon (float): fraction of the time to explore
     arms (list): collection of bandits to pull
    """
    def __init__(self, epsilon, arms):
        self.epsilon = epsilon
        self.arms = arms
        self._counts = None
        self._rewards = None
        return
    @property
    def best_arm(self):
        """Index of the arm with the most reward"""
        index = self.rewards.max()
        return find_first(index, self.rewards)
    @property
    def counts(self):
        """counts of times each arm is pulled

        Returns:
         numpy.array: array of counts
        """
        if self._counts is None:
            self._counts = numpy.zeros(len(self.arms))
        return self._counts
    @property
    def rewards(self):
        """array of running average of rewards for each arms

        Returns:
         numpy.array: running averages
        """
        if self._rewards is None:
            self._rewards = numpy.zeros(len(self.arms))
        return self._rewards
