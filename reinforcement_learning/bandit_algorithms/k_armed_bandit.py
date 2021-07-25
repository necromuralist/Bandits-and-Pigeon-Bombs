# python
import random

# pypi
import numpy


class Arm:
    """An arm for the bandit"""
    def __init__(self):
        self._center = None
        return

    @property
    def center(self) -> float:
        """The center of the payout distribution"""
        if self._center is None:
            self._center = random.gauss(mu=0, sigma=1)
        return self._center

    def pull(self) -> float:
        """Get the payout for pulling this arm

        Returns:
         reward for this pull
        """
        return random.gauss(mu=self.center, sigma=1)


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
            self._best_arm = numpy.array([
                arm.center for arm in self.arms]).argmax()
        return self._best_arm

    def __call__(self, arm: int) -> float:
        """Pulls the arm

        Args:
         arm: the index for the arm to pull
        
        Returns:
         the payout from the arm
        """
        return self.arms[arm].pull()


class EpsilonExplorer:
    """runs the epsilon-greedy algorithm

    Args:
     epsilon: fraction of the time to explore
     arms: number of arms for the bandit
     steps: number of steps to run the algorithm
    """
    def __init__(self, epsilon: float, arms: int, steps: int):
        self.epsilon = epsilon
        self.arms = arms
        self.steps = steps

        self._expected_reward = None
        self._pulled = None
        self._bandit = None
        self._rewards = None
        self._is_optimal = None
        return

    @property
    def expected_reward(self) -> numpy.ndarray:
        """The expected reward for each arm"""
        if self._expected_reward is None:
            self._expected_reward = numpy.zeros(self.arms)
        return self._expected_reward

    @property
    def pulled(self) -> numpy.ndarray:
        """Number of times each arm has been pulled"""
        if self._pulled is None:
            self._pulled = numpy.zeros(self.arms)
        return self._pulled

    @property
    def bandit(self) -> Bandit:
        """k-armed bandit"""
        if self._bandit is None:
            self._bandit = Bandit(k=self.arms)
        return self._bandit

    @property
    def rewards(self) -> numpy.ndarray:
        """The reward for each step"""
        if self._rewards is None:
            self._rewards = numpy.zeros(self.steps)
        return self._rewards

    @property
    def is_optimal(self) -> numpy.ndarray:
        """Track which steps pulled the optimal arm"""
        if self._is_optimal is None:
            self._is_optimal = numpy.zeros(self.steps)
        return self._is_optimal

    def __call__(self):
        """Runs the epsilon-greedy algorithm"""
        for step in range(self.steps):
            exploit = random.random()
            arm = (self.expected_reward.argmax() if exploit > self.epsilon
                   else random.randrange(self.arms))
            reward = self.bandit(arm)
            self.pulled[arm] += 1
            previous_expected = self.expected_reward[arm]
            self.expected_reward[arm] = (
                previous_expected +
                (reward - previous_expected)/self.pulled[arm])
            self.rewards[step] = reward
            self.is_optimal[step] = int(arm == self.bandit.best_arm)
        return
