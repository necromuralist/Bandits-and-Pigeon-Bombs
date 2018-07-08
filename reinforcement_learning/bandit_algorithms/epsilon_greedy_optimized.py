# python
import random

# pypi
from numba import (
    jit,
    jitclass,
    )
import numba
import numpy

spec = [
    ("epsilon", numba.double),
    ("arms", numba.double[:]),
    ("counts", numba.double[:]),
    ("rewards", numba.double[:]),
    ("total_reward", numba.int64),
]

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
        
@jitclass(spec)
class EpsilonGreedyOptimized(object):
    """The Epsilon Greedy Algorithm

    Args:
     epsilon (float): fraction of the time to explore
     arms (list): collection of probabilities for bandit arm
    """
    def __init__(self, epsilon, arms):
        self.epsilon = epsilon
        self.arms = arms
        self.counts = numpy.zeros(len(arms))
        self.rewards = numpy.zeros(len(arms))
        self.total_reward = 0
        return

    @property
    def best_arm(self):
        """Index of the arm with the most reward"""
        index = self.rewards.max()
        return find_first(index, self.rewards)

    def select_arm(self):
        """chooses the next arm to update
    
        Returns:
         int: index of the next arm to pull
        """
        if random.random() < self.epsilon:
            return random.randrange(len(self.arms))
        return self.best_arm

    def pull_arm(self, arm):
        """gets the reward
        
        Args:
         arm (int): index for the arm-probability array
        Returns:
         int: reward or no reward
        """
        if random.random() > self.arms[arm]:
            return 0
        return 1

    def update(self, arm):
        """pulls the arm and updates the value
    
        Args:
         arm (int): index of the arm to pull
        """
        self.counts[arm] += 1
        count = self.counts[arm]
        average_reward = self.rewards[arm]
        reward = self.pull_arm(arm)
        self.total_reward += reward
        self.rewards[arm] = (((count - 1)/float(count)) * average_reward
                            + (reward/float(count)))
        return

    def reset(self):
        """sets the counts, rewards, total_reward to 0s
    
        This lets you re-used the EpsilonGreedy
        """
        self.counts = numpy.zeros(len(self.arms))
        self.rewards = numpy.zeros(len(self.arms))
        self.total_reward = 0
        return

    def __call__(self):
        """chooses an arm and updates the rewards"""
        arm = self.select_arm()
        self.update(arm)
        return
