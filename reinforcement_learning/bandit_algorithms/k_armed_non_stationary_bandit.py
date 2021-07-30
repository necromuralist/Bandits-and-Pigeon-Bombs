# python
import random

# pypi
import numpy


class RandomWalkArm:
    """A random-walk arm

    Args:
     reward: the starting reward
     center: the center of the reward increment distribution
     sigma: the standard deviation for the reward increment
    """
    def __init__(self, reward: float=0, center: float=0, sigma: float=0.01):
        self.reward = reward
        self.center = center
        self.sigma = sigma
        return

    def update(self) -> None:
        """Update the reward"""
        self.reward += random.gauss(mu=self.center, sigma=self.sigma)
        return

    def pull(self) -> float:
        """Pull the arm and get a reward

        Returns:
         the reward for pulling this arm
        """
        return self.reward


class MovingBandit:
    """Bandit that uses the random-walk arms

    Args:
     k: number of arms for the bandit
     starting_reward: reward to start the arms off with
     center: the center of the probability distribution to update arms
     sigma: the spread of the distribution to update arms
    """
    def __init__(self, k: int=10, starting_reward: int=0,
                 center: float=0,
                 sigma: float=0.01) -> None:
        self.k = k
        self.starting_reward = starting_reward
        self.center = center
        self.sigma = sigma
        self._arms = None
        return

    @property
    def arms(self) -> list:
        """The arms of the bandit"""
        if self._arms is None:
            self._arms = [RandomWalkArm(
                reward=self.starting_reward,
                center=self.center,
                sigma=self.sigma) for arm in range(self.k)]
        return self._arms

    @property
    def best_arm(self) -> int:
        """the current best arm

        Returns:
         index of the arm with the highest reward
        """
        return numpy.array([arm.reward
                            for arm in self.arms]).argmax()

    def reset(self) -> None:
        """Resets the arms"""
        self._arms = None
        return

    def __call__(self, arm: int) -> float:
        """Updates the arms and returns the reward for the given arm

        Args:
         arm: index of the arm to return the reward from

        Returns:
         the reward for the given arm
        """
        self._best_arm = None
        for _arm in self.arms:
            _arm.update()
        return self.arms[arm].reward


class WalkExplorer:
    """Epsilon-Greedy explorer with constant alpha

    Args:
     epsilon: fraction of the time to explore
     alpha: learning rate for updating the expected reward
     arms: number of arms for the bandit
     steps: number of times to pull the bandit's arm
     starting_reward: the starting reward for each arm
     center: center of the reward update distribution
     sigma: spread of the reward update distribution
    """
    def __init__(self, epsilon: float=0.1, alpha: float=0.1,
                 arms: int=10, steps: int=10000,
                 starting_reward: float=0, center: float=0, sigma=0.01):
        self.epsilon = epsilon
        self.alpha = alpha
        self.arms = arms
        self.steps = steps
        self.starting_reward = starting_reward
        self.center = center
        self.sigma = sigma

        self._expected_reward = None
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
    def bandit(self) -> MovingBandit:
        """The multi-armed bandit"""
        if self._bandit is None:
            self._bandit = MovingBandit(
                k=self.arms,
                starting_reward=self.starting_reward,
                center=self.center, sigma=self.sigma)
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

    def reset(self) -> None:
        """Resets the arrays and the bandit"""
        self._is_optimal = None
        self._rewards = None
        self.bandit.reset()
        self._expected_reward = None
        return

    def __call__(self):
        """Runs the epsilon-greed algorithm with constant alpha"""
        for step in range(self.steps):
            exploit = random.random()
            arm = (self.expected_reward.argmax() if exploit > self.epsilon
                   else random.randrange(self.arms))
            reward = self.bandit(arm)
            previous_expected = self.expected_reward[arm]
            self.expected_reward[arm] = (
                previous_expected +
                self.alpha * (reward - previous_expected))
            self.rewards[step] = reward
            self.is_optimal[step] = int(arm == self.bandit.best_arm)
        return