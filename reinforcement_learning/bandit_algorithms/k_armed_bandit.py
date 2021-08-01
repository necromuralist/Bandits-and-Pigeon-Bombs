# python
import random

# pypi
import numpy

numpy_random = numpy.random.default_rng()


class Arm:
    """An arm for the bandit

    Args:
     center: the mean of the reward distribution
     sigma: the spread of the reward distribution
    """
    def __init__(self, center: float=None, sigma: float=1):
        self._center = center
        self.sigma = sigma
        return

    @property
    def center(self) -> float:
        """The center of the payout distribution"""
        if self._center is None:
            self._center = random.gauss(mu=0, sigma=self.sigma)
        return self._center

    def pull(self) -> float:
        """Get the payout for pulling this arm

        Returns:
         reward for this pull
        """
        return random.gauss(mu=self.center, sigma=self.sigma)


class Bandit:
    """A k-armed bandit

    Args:
     k: number of arms for the bandit
    """
    def __init__(self, k: int=10):
        self.k = k
        self._arms = None
        self._best_arm = None
        return

    @property
    def arms(self) -> list:
        """The arms for the bandit"""
        if self._arms is None:
            self._arms = [Arm() for arm in range(self.k)]
        return self._arms

    @property
    def best_arm(self) -> int:
        """The arm with the highest mean

        Returns:
         index of the arm with the highest mean payoff
        """
        if self._best_arm is None:
            centers = numpy.array([
                arm.center for arm in self.arms])
            highest = numpy.amax(centers)
            best = numpy.where(centers == highest)[0]
            self._best_arm = numpy_random.choice(best)
        return self._best_arm

    def reset(self):
        """Resets the arms and best arm"""
        self._arms = None
        self._best_arm = None
        return

    def __call__(self, arm: int) -> float:
        """Pulls the arm

        Args:
         arm: the index for the arm to pull
        
        Returns:
         the payout from the arm
        """
        return self.arms[arm].pull()


class AverageMemory:
    """The average sample memory for the epsilon greedy agent

    Args:
     arms: number of arms on the bandit
    """
    def __init__(self, arms: int):
        self.arms = arms
        self._pulled = None
        self._expected_reward = None        
        return

    @property
    def pulled(self) -> numpy.ndarray:
        """Count of how many times each arm was pulled"""
        if self._pulled is None:
            self._pulled = numpy.zeros(self.arms)
        return self._pulled

    @property
    def expected_reward(self) -> numpy.ndarray:
        """The expected reward for each arm"""
        if self._expected_reward is None:
            self._expected_reward = numpy.zeros(self.arms)
        return self._expected_reward

    @property
    def best_arm(self) -> int:
        """The index of the best arm"""
        best = numpy.amax(self.expected_reward)
        bestest = [index for index in range(len(self.expected_reward))
                   if self.expected_reward[index] == best]
        return numpy.random.choice(bestest)

    @property
    def random_arm(self) -> int:
        """Index of a random arm"""
        return numpy_random.integers(self.arms)

    def update(self, arm: int, reward: float) -> None:
        """Updates the expected reward

        Args:
         arm: the arm that was pulled to earn the reward
         reward: the reward earned by pulling the arm
        """
        self.pulled[arm] += 1
        expected = self.expected_reward[arm]
        self.expected_reward[arm] = expected + (reward - expected)/self.pulled[arm]
        return


class EpsilonExplorer:
    """runs the epsilon-greedy algorithm

    Args:
     epsilon: fraction of the time to explore
     arms: number of arms for the bandit
    """
    def __init__(self, epsilon: float, arms: int):
        self.epsilon = epsilon
        self.arms = arms
        self._memory = None
        return

    @property
    def memory(self) -> AverageMemory:
        """The memory of rewards earned"""
        if self._memory is None:
            self._memory = AverageMemory(arms = self.arms)
        return self._memory

    @property
    def first_arm(self) -> int:
        """The first arm to use"""
        self.most_recent_arm = self.memory.best_arm
        return self.most_recent_arm

    def reset(self):
        """Resets the memory"""
        self._memory = None
        return

    def __call__(self, reward: float) -> int:
        """Runs the epsilon-greedy algorithm

        Args:
         reward: the reward from the bandit

        Returns:
         the next arm to pull
        """
        self.memory.update(self.most_recent_arm, reward)
        exploit = random.random()
        self.most_recent_arm = (
            self.memory.best_arm if exploit > self.epsilon
            else self.memory.random_arm)
        return self.most_recent_arm
