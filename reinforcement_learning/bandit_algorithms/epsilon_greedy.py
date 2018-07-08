# python
import random

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
        self.total_reward = None
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
            self._counts = numpy.zeros(len(self.arms), dtype=int)
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

    def select_arm(self):
        """chooses the next arm to update
    
        Returns:
         int: index of the next arm to pull
        """
        if random.random() < self.epsilon:
            return random.randrange(len(self.arms))
        return self.best_arm

    def update(self, arm):
        """pulls the arm and updates the value
    
        Args:
         arm (int): index of the arm to pull
        """
        self.counts[arm] += 1
        count = self.counts[arm]
        average_reward = self.rewards[arm]
        reward = self.arms[arm]()
        self.total_reward += reward
        self.rewards[arm] = (((count - 1)/float(count)) * average_reward
                            + (reward/float(count)))
        return

    def reset(self):
        """sets the counts and rewards to None
    
        This lets you re-used the EpsilonGreedy without re-constructing
        the arms
        """
        self._counts = None
        self._rewards = None
        self.total_reward = 0
        return

    def __call__(self):
        """chooses an arm and updates the rewards"""
        arm = self.select_arm()
        self.update(arm)
        return
