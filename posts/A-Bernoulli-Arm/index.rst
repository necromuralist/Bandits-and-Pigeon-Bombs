.. title: A Bernoulli Arm
.. slug: A-Bernoulli-Arm
.. date: 2017-07-30 18:37:00
.. tags: algorithm
.. link: 
.. description: Implementation of a Bandit Arm to test the reinforcement algorithms.
.. type: text
.. author: hades



1 Introduction
--------------

This is an implementation of one arm of a `n-armed bandit <https://en.wikipedia.org/wiki/Multi-armed_bandit>`_ to test the Epsilon Greedy algorithm. It takes a probability that it will return a reward. It also optionally let's you set the penalty and reward values, but defaults to a reward of 1 and a penalty of 0 (so it's really no reward more than a penalty).

2 Imports
---------

.. code:: python

    # python standard library
    import random

3 Bernoulli Arm
---------------

The Bernoulli Arm will generate a value when its arm is pulled at a payout rate specified by the \`probability\` value.

.. code:: python

    <<imports>> 

    class BernoulliArm(object):
        """A simulation of one arm of a multi-armed bandit
    
        Args:
         probability (float): probability of a reward
         reward (float): value to return on a win
         penalty (float): value to return on a loss
        """
        <<constructor>>

        <<call>>

3.1 Constructor
~~~~~~~~~~~~~~~

The constructor takes three values:

- probability of winning

- reward on winning

- penalty on losing

Because of the way the problem is set up, the reward and penalty are already set at 1 and 0, but I didn't want there to be magic numbers so they can be changed if needed.

.. code:: python

    def __init__(self, probability, reward=1, penalty=0):
        self.probability = probability
        self.reward = reward
        self.penalty = penalty
        return

3.2 The Call
~~~~~~~~~~~~

This is called ``pull`` in most cases, but I thought it would be more uniform to put it in a call.

.. code:: python

    def __call__(self):
        """pulls the arm and returns a reward or penalty

        Returns:
         float: value returned on pulling the arm
        """
        if random.random() > self.probability:
            return self.penalty
        return self.reward
