.. title: Epsilon Greedy with Normal-Distribution Payouts
.. slug: Epsilon-Greedy-with-Normal-Distribution-Payouts
.. date: 2017-08-01 18:48
.. tags: bandits reinforcementLearning
.. link: 
.. description: The Epsilon Greedy with a Gaussian arm.
.. type: text
.. author: Brunhilde



1 Epsilon Greedy
----------------

Since the Optimistic Initial Values agent can't use the Bernoulli Arm, I'm creating a version of the Epsilon Greedy Optimized that expects the ``arms`` to be the population-mean for their payouts and the ``pull_arm`` will return a set of normally-distributed around that mean.

.. code:: python

    <<optimized-imports>>

    <<spec>>

    <<find-first>>
    @jitclass(spec)
    class EpsilonGreedyNormal(object):
        """The Epsilon Greedy Algorithm With Normal Arm

        Args:
         epsilon (float): fraction of the time to explore
         arms (list): collection of probabilities for bandit arm
        """
        <<optimized-constructor>>

        <<best-arm>>

        <<select-arm>>

        <<optimized-pull-arm>>

        <<optimized-update>>

        <<optimized-reset>>

        <<call>>

1.1 Optimized Imports
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # pypi
    from numba import (
        jit,
        jitclass,
        )
    import numba
    import numpy

1.2 The Spec
~~~~~~~~~~~~

This is how you tell numba what attributes the class will have.

.. code:: python

    spec = [
        ("epsilon", numba.double),
        ("arms", numba.double[:]),
        ("counts", numba.double[:]),
        ("rewards", numba.double[:]),
        ("total_reward", numba.int64),
    ]

1.3 The Constructor
~~~~~~~~~~~~~~~~~~~

The constructor takes two arguments - *epsilon* and *arms*. The *arms* list should contain the mean payout for each arm.

.. code:: python

    def __init__(self, epsilon, arms):
        self.epsilon = epsilon
        self.arms = arms
        self.counts = numpy.zeros(len(arms))
        self.rewards = numpy.zeros(len(arms))
        self.total_reward = 0
        return

1.4 Reset
~~~~~~~~~

.. code:: python

    def reset(self):
        """sets the counts, rewards, total_reward to 0s

        This lets you re-used the EpsilonGreedy
        """
        self.counts = numpy.zeros(len(self.arms))
        self.rewards = numpy.zeros(len(self.arms))
        self.total_reward = 0
        return

1.5 Best Arm
~~~~~~~~~~~~

The ``best_arm`` property returns the index of the arm that has the highest average reward so far. It returns the index instead of the arm itself because it's used to get the matching counts and rewards in the ``update`` method. Since I'm using the ``jitclass`` decorator I'm going to get rid of ``first_find``.

.. code:: python

    @property
    def best_arm(self):
        """Index of the arm with the most reward"""
        item = self.rewards.max()
        for index in range(len(self.rewards)):
            if item == self.rewards[index]:
                return index
        return

1.6 Select Arm
~~~~~~~~~~~~~~

This differs from the other Epsilon Greedy code only in that I'm using numpy instead of python for the random function.

.. code:: python

    def select_arm(self):
        """chooses the next arm to update

        Returns:
         int: index of the next arm to pull
        """
        if numpy.random.random() < self.epsilon:
            return numpy.random.randint(len(self.arms))
        return self.best_arm

1.7 Pull Arm
~~~~~~~~~~~~

Since we can't give user-defined objects as attributes of the class, this version will be both algorithm and bandit. This is what's different from the other Epsilon Greedy algorithms in that we're returning the arm's mean plus a random number from the normal distribution. If numba allowed us to pass in objects maybe we could have just switched out bandits. I need to look into how to make that work.

.. code:: python

    def pull_arm(self, arm):
        """gets the reward
    
        Args:
         arm (int): index for the arm-probability array
        Returns:
         float: reward
        """
        return numpy.random.randn() + self.arms[arm]

1.8 Update
~~~~~~~~~~

The update method pulls the arm whose index it is given and then updates the count and reward. Here we're calling the ``pull_arm`` method instead of using a ``BernoulliArm`` so we can't re-use the original method.

.. code:: python

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
