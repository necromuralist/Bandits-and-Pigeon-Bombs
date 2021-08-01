# python
from functools import partial

import random

# pypi
from expects import (
    equal,
    expect
)

from pytest_bdd import (
    given,
    then,
    when,
)

import pytest_bdd

from .fixtures import katamari

# code under test
from reinforcement_learning.bandit_algorithms.k_armed_bandit import EpsilonExplorer


scenario = partial(pytest_bdd.scenario, "../features/epsilon_greedy.feature")

# ********** check setup
@scenario("Preparing to run the explorer")
def test_preparing_to_run_the_explorer():
    return


@given('an Epsilon Explorer')
def an_epsilon_explorer(katamari):
    katamari.epsilon = random.random()
    katamari.arms = random.randrange(1, 100)
    katamari.explorer = EpsilonExplorer(epsilon=katamari.epsilon,
                                        arms=katamari.arms)
    return


@when('the properties are checked')
def the_properties_are_checked(katamari):
    return


@then('they are all zero')
def they_are_all_zero(katamari):
    ZERO = 0
    return

@then('they have the right length')
def they_have_the_right_length(katamari):
    return

