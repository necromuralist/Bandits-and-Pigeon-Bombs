# pypi
import numpy

# this code
from reinforcement_learning.bandit_algorithms.k_armed_bandit import Bandit, EpsilonExplorer



class TestBed:
    """Testbed for the epsilon-greedy bandit optimizer

    Args:
     epsilon: fraction of the time for the explorer to explore
     arms: number of arms for the bandit
     runs: number of times to run the explorer
     steps: number of steps for the explorer to take (time)
     explorer: epsilon explorer instance
     reporting_interval: how often to report what run you're on
    """
    def __init__(self, epsilon: float, arms: int=10, runs: int=2000,
                 steps: int=1000, explorer: type=None, reporting_interval: int=100):
        self.epsilon = epsilon
        self.arms = arms
        self.runs = runs
        self.steps = steps
        self.reporting_interval = reporting_interval
        self._bandit = None
        self._total_rewards = None
        self._optimal_choices = None
        self._explore = explorer
        return

    @property
    def bandit(self) -> Bandit:
        """The K-armed bandit"""
        if self._bandit is None:
            self._bandit = Bandit(self.arms)
        return self._bandit

    @property
    def total_rewards(self) -> numpy.ndarray:
        """The total rewards earned from the bandit"""
        if self._total_rewards is None:
            self._total_rewards = numpy.zeros(self.steps)
        return self._total_rewards

    @total_rewards.setter
    def total_rewards(self, new_rewards: numpy.ndarray):
        """Sets the total_rewards

        Args:
         new_rewards: updated rewards
        """
        self._total_rewards = new_rewards
        return

    @property
    def optimal_choices(self) -> numpy.ndarray:
        """The number of times the choice made was the optimal arm"""
        if self._optimal_choices is None:
            self._optimal_choices = numpy.zeros(self.steps)
        return self._optimal_choices

    @optimal_choices.setter
    def optimal_choices(self, new_optimal_count: numpy.ndarray):
        """Sets the optimal choices

        Args:
         new_optimal_count: updated optimal_choices
        """
        self._optimal_choices = new_optimal_count
        return

    @property
    def explore(self) -> EpsilonExplorer:
        """The epsilon greedy explorer"""
        if self._explore is None:
            self._explore = EpsilonExplorer(epsilon=self.epsilon,
                                             arms=self.arms)
        return self._explore

    def __call__(self):
        """Runs the explorer"""
        for run in range(self.runs):
            if run % self.reporting_interval == 0:
                print(f"(epsilon={self.epsilon}) Run {run}")
            arm = self.explore.first_arm
            rewards = numpy.zeros(self.steps)
            is_optimal = numpy.zeros(self.steps)
            for step in range(self.steps):
                reward = self.bandit(arm)
                rewards[step] = reward
                is_optimal[step] = int(arm == self.bandit.best_arm)
                arm = self.explore(reward=reward)
            self.explore.reset()
            self.bandit.reset()
            self.total_rewards += rewards
            self.optimal_choices += is_optimal
        return
