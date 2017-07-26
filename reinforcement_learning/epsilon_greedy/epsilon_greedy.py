# pypi
import numpy

class EpsilonGreedy(object):
    """The Epsilon Greedy Algorithm

    Args:
     epsilon (float): fraction of the time to explore
     arms (list): collection of bandits to pull
    """
    def __init__(self, epsilon, arms):
        self.epsilon = epsilon
        self.arms = arms
        self._count = None
        self._rewards = None
        return
    @property
    def best_arm(self):
        """Index of the arm with the most reward"""
        index = max(self.rewards)        
        return self.rewards.index(index)

    @property
    def rewards(self):
        """array of running average of rewards for each arms

        Returns:
         numpy.array: running averages
        """
        if self._rewards is None:
            self._rewards = numpy.zeros(len(self.arms))
        return self._rewards
