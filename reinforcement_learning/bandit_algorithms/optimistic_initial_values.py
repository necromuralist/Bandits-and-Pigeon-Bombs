from numba import jitclass
import numba
import numpy

SPEC = [
    ("arms", numba.double[:]),
    ("counts", numba.double[:]),
    ("rewards", numba.double[:]),
    ("total_reward", numba.int64),
    ("initial_reward", numba.double),
]

@jitclass(SPEC)
class OptimisticInitialValues(object):
    """Optimistic Initial Values greedy algorithm

    Args:
     numpy.array[float]: payout-probabilities for each arm
    """    

    def __init__(self, arms, initial_reward):
        self.arms = arms
        self.counts = numpy.zeros(len(arms))
        self.rewards = numpy.zeros(len(arms)) + initial_reward
        self.total_reward = 0
        self.initial_reward = initial_reward
        return
    
    def select_arm(self):
        """Index of the arm with the most reward
    
        Returns:
         integer: index of arm with highest average reward
        """
        item = self.rewards.max()
        for index in range(len(self.rewards)):
            if item == self.rewards[index]:
                return index
    
    def pull_arm(self, arm):
        """gets the reward
            
        Args:
         arm (int): index for the arm population-mean array
        Returns:
         float: payout for the arm
        """
        return numpy.random.randn() + self.arms[arm]
    
    def update(self, arm):
        """pulls the arm and updates the average reward
        
        also updates the total_reward the algorithm has earned so far
        
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
        self.rewards = numpy.zeros(len(self.arms)) + self.initial_reward
        self.total_reward = 0
        return
