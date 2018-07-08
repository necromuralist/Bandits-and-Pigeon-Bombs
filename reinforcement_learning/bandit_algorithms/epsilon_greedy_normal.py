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


@jitclass(spec)
class EpsilonGreedyNormal(object):
    """The Epsilon Greedy Algorithm With Normal Arm

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
        item = self.rewards.max()
        for index in range(len(self.rewards)):
            if item == self.rewards[index]:
                return index
        return

    def select_arm(self):
        """chooses the next arm to update
    
        Returns:
         int: index of the next arm to pull
        """
        if numpy.random.random() < self.epsilon:
            return numpy.random.randint(len(self.arms))
        return self.best_arm

    def pull_arm(self, arm):
        """gets the reward
        
        Args:
         arm (int): index for the arm-probability array
        Returns:
         float: reward
        """
        return numpy.random.randn() + self.arms[arm]

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
