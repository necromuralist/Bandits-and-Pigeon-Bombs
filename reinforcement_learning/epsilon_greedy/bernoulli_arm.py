# python standard library
import random 

class BernoulliArm(object):
    """A simulation of one arm of a multi-armed bandit
    
    Args:
     probability (float): probability of a reward
     reward (float): value to return on a win
     penalty (float): value to return on a loss
    """
    def __init__(self, probability, reward=1, penalty=0):
        self.probability = probability
        self.reward = reward
        self.penalty = penalty
        return

    def __call__(self):
        """pulls the arm and returns a reward or penalty
    
        Returns:
         float: value returned on pulling the arm
        """
        if random.random() > self.probability:
            return self.penalty
        return self.reward
