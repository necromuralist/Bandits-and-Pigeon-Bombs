.. title: Optimistic Initial Values
.. slug: Optimistic-Initial-Values
.. date: 2017-08-01 18:47
.. tags: bandits reinforcementLearning
.. link: 
.. description: The Optimistic Initial Values agent.
.. type: text
.. author: Brunhilde



1 Introduction
--------------

This is one possible to the n-armed bandit problem. It is similar to the *Epsilon Greedy* algorithm except that instead of using a conditional to decide whether to explore or exploit, the algorithm sets the estimated (mean) payout for each arm to 1 (the theoretical maximum for our case) and then always exploits. As things proceed, the arms will settle down to their actual payoff-rates and those that haven't been explored will be chosen because they are still too high.

2 The Tangle
------------

This is the no-web template to build the final file.

.. code:: python

    <<imports>>

    <<spec>>

    <<class-declaration>>

        <<constructor>>
    
        <<select-arm>>
    
        <<pull-arm>>
    
        <<update-arm>>
    
        <<reset>>

3 Imports
---------

These are our external dependencies.

.. code:: python

    from numba import jitclass
    import numba
    import numpy

4 The Spec
----------

In order to use numba with the ``OptimisticInitialValues`` class you have to create a 'spec' that tells numba what the data-types are for each of its fields.

.. code:: python

    SPEC = [
        ("arms", numba.double[:]),
        ("counts", numba.double[:]),
        ("rewards", numba.double[:]),
        ("total_reward", numba.int64),
        ("initial_reward", numba.double),
    ]

5 The Class Declaration
-----------------------

.. code:: python

    @jitclass(SPEC)
    class OptimisticInitialValues(object):
        """Optimistic Initial Values greedy algorithm

        Args:
         numpy.array[float]: payout-probabilities for each arm
        """    

6 The Constructor
-----------------

Here's our first change from the epsilon-greedy algorithm. We no longer have an ``epsilon`` value and instead of initializing the ``rewards`` as zeros we initialize them with an 'initial' reward. Also, although you can't see it here, the arms have to be a list of mean payout values (see the ``pull_arm`` method below).

.. code:: python

    def __init__(self, arms, initial_reward):
        self.arms = arms
        self.counts = numpy.zeros(len(arms))
        self.rewards = numpy.zeros(len(arms)) + initial_reward
        self.total_reward = 0
        self.initial_reward = initial_reward
        return

7 Select Arm
------------

This chooses the next arm. Unlike the epsilon-greedy algorithm it will always pick the 'best' arm, choosing the first if there is a tie. Since the whole class is in the jit I'm also not using the external ``find_first`` method.

.. code:: python

    def select_arm(self):
        """Index of the arm with the most reward

        Returns:
         integer: index of arm with highest average reward
        """
        item = self.rewards.max()
        for index in range(len(self.rewards)):
            if item == self.rewards[index]:
                return index

8 Pull Arm
----------

This gets the reward for the arm. with a Bernoulli arm, there's a chance that an arm will be set to 0 on its first pull, at which point you will never explore it (since there's no exploration), so even the best arm might get wiped out. To fix this you need a different scheme. This one uses a population mean (selected ``from self.arms``) which has noise added by selecting from the standard normal distribution.

.. code:: python

    def pull_arm(self, arm):
        """gets the reward
        
        Args:
         arm (int): index for the arm population-mean array
        Returns:
         float: payout for the arm
        """
        return numpy.random.randn() + self.arms[arm]

9 Update Arm
------------

This pulls the arm and updates the reward. This works the same as the ``epsilon-greedy`` version does.

.. code:: python

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

10 Reset
--------

This resets the values so that you can re-use the algorithm. As with the constructor, it sets the ``rewards`` to all ones instead of zeros as was the case with the epsilon-greedy algorithm.

.. code:: python

    def reset(self):
        """sets the counts, rewards, total_reward to 0s
    
        This lets you re-used the EpsilonGreedy
        """
        self.counts = numpy.zeros(len(self.arms))
        self.rewards = numpy.zeros(len(self.arms)) + self.initial_reward
        self.total_reward = 0
        return
