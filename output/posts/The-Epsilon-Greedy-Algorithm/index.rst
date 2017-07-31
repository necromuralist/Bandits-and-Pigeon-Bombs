.. title: The Epsilon Greedy Algorithm
.. slug: The-Epsilon-Greedy-Algorithm
.. date: 2017-07-30 18:22:00
.. tags: algorithm reinforcementlearning
.. link: 
.. description: Epsilon Greedy Reinforcement Algorithm
.. type: text
.. author: hades



1 Background
------------

This is an implementation of the Epsilon Greedy algorithm to find solutions for the multi-arm-bandit problem.

2 Imports
---------

.. code:: python

    # python
    import random

    # pypi
    from numba import jit
    import numpy

3 Find First
------------

This is a helper function to find the first matching item in an array-like collection.

.. code:: python

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

4 Epsilon Greedy
----------------

The *epsilon-greedy* algorithm tries to solve the exploitation-exploration dilemna by exploring a fraction of the time (set by *epsilon*) and using the best solution found so far the rest of the time. This implementation is based on the one in Bandit Algorithms for Website Optimization [1]_ .

.. code:: python

    <<imports>>

    <<find-first>>
    class EpsilonGreedy(object):
        """The Epsilon Greedy Algorithm

        Args:
         epsilon (float): fraction of the time to explore
         arms (list): collection of bandits to pull
        """
        <<constructor>>

        <<best-arm>>

        <<counts>>

        <<rewards>>

        <<select-arm>>

        <<update>>

        <<reset>>

        <<call>>

4.1 The Constructor
~~~~~~~~~~~~~~~~~~~

The constructor takes two arguments - *epsilon* and *arms*. The *arms* list should contain bandits that return a reward or penalty when pulled (called).

.. code:: python

    def __init__(self, epsilon, arms):
        self.epsilon = epsilon
        self.arms = arms
        self._counts = None
        self._rewards = None
        self.total_reward = None
        return

4.2 Best Arm
~~~~~~~~~~~~

The ``best_arm`` property returns the index of the arm that has the highest average reward so far. It returns the index instead of the arm itself because it's used to get the matching counts and rewards in the ``update`` method.

.. code:: python

    @property
    def best_arm(self):
        """Index of the arm with the most reward"""
        index = self.rewards.max()
        return find_first(index, self.rewards)

4.3 Counts
~~~~~~~~~~

The \`counts\` keeps track of the number of times each arm is pulled.

.. code:: python

    @property
    def counts(self):
        """counts of times each arm is pulled

        Returns:
         numpy.array: array of counts
        """
        if self._counts is None:
            self._counts = numpy.zeros(len(self.arms), dtype=int)
        return self._counts

4.4 Rewards
~~~~~~~~~~~

The ``rewards`` attributes holds the running average reward that each arm has returned.

.. code:: python

    @property
    def rewards(self):
        """array of running average of rewards for each arms

        Returns:
         numpy.array: running averages
        """
        if self._rewards is None:
            self._rewards = numpy.zeros(len(self.arms))
        return self._rewards

4.5 Reset
~~~~~~~~~

.. code:: python

    def reset(self):
        """sets the counts and rewards to None

        This lets you re-used the EpsilonGreedy without re-constructing
        the arms
        """
        self._counts = None
        self._rewards = None
        self.total_reward = 0
        return

4.6 Select Arm
~~~~~~~~~~~~~~

The *select\_arm* method will choose either the best arm or a random one based on a randomly drawn value and how it compares to epsilon.

.. code:: python

    def select_arm(self):
        """chooses the next arm to update

        Returns:
         int: index of the next arm to pull
        """
        if random.random() < self.epsilon:
            return random.randrange(len(self.arms))
        return self.best_arm

4.7 Update
~~~~~~~~~~

The update method pulls the arm whose index it is given and then updates the count and reward.

.. code:: python

    def update(self, arm):
        """pulls the arm and updates the value

        Args:
         arm (int): index of the arm to pull
        """
        self.counts[arm] += 1
        count = self.counts[arm]
        average_reward = self.rewards[arm]
        reward = self.arms[arm]()
        self.total_reward += reward
        self.rewards[arm] = (((count - 1)/float(count)) * average_reward
                            + (reward/float(count)))
        return

4.8 Call
~~~~~~~~

The *\_\_call\_\_* method will be the main update method that unifies the naming conventions found in the books.

.. code:: python

    def __call__(self):
        """chooses an arm and updates the rewards"""
        arm = self.select_arm()
        self.update(arm)
        return

5 Epsilon Greedy Optimized
--------------------------

It turns out that while the implementation above works correctly, it can be rather slow, given that we need to train it thousands of times to get meaningful results. This is a numba-compatible version that drops the testing time from around 11 minutes to a minute or less. One of the restrictions of using classes in numba is that you have to declare the types of all the attributes of the class (this happens in the `spec` passed to the `jitclass` decorator). This means that I can't pass in `BernoulliArm` objects to the constructor, because `numba` has no idea what they are, so this solution is a hybrid greedy algorithm and bandit arm mashed together.

The documentation for `numba` states that you have to initialize the attributes in the `__init__` method so I'm getting rid of the properties that build the numpy arrays and moving their creation to the constructor. In addition, the code that no longer expects the =BernoulliArm= objects will have to be re-implemented. In the tangle code anything with the `optimized-` prefix is re-implemented (other than the `spec`), otherwise the code is being pulled in from the original `EpsilonGreedy` implementation.

.. code:: python

    <<optimized-imports>>

    <<spec>>

    <<find-first>>
    @jitclass(spec)
    class EpsilonGreedyOptimized(object):
        """The Epsilon Greedy Algorithm

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

5.1 Optimized Imports
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # python
    import random

    # pypi
    from numba import (
        jit,
        jitclass,
        )
    import numba
    import numpy

5.2 The Spec
~~~~~~~~~~~~

This is how you tell numba what attributes the class will have. This is where most of the errors were when I first tried this. The error-messages aren't particularly helpful. Just be aware that this is the first place you should look if things crash.

.. code:: python

    spec = [
        ("epsilon", numba.double),
        ("arms", numba.double[:]),
        ("counts", numba.double[:]),
        ("rewards", numba.double[:]),
        ("total_reward", numba.int64),
    ]

5.3 The Constructor
~~~~~~~~~~~~~~~~~~~

The constructor takes two arguments - *epsilon* and *arms*. The *arms* list should contain probabilities that a reward or penalty will be returned when pulled.

.. code:: python

    def __init__(self, epsilon, arms):
        self.epsilon = epsilon
        self.arms = arms
        self.counts = numpy.zeros(len(arms))
        self.rewards = numpy.zeros(len(arms))
        self.total_reward = 0
        return

5.4 Reset
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

5.5 Pull Arm
~~~~~~~~~~~~

Since we can't give user-defined objects as attributes of the class, this version will be both algorithm and bandit.

.. code:: python

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

5.6 Update
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

6 References
------------


.. [1] Bandit Algorithms for Website Optimization by John Myles White. Copyright 2013 John Myles White, 978-1-449-34133-6
