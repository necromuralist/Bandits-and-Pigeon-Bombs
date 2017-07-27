# coding=utf-8
"""Epsilon Greedy Implementation feature tests."""
# python standard library
from functools import partial
import random

# pypi
from expects import (
    be,
    be_true,
    equal,
    expect,
)
from pytest_bdd import (
    given,
    then,
    when,
)
import numpy
import pytest_bdd

# testing
from .fixtures import context
from ..epsilon_greedy import EpsilonGreedy

scenario = partial(pytest_bdd.scenario, "epsilon_greedy.feature")
and_also = then
And = when
and_given = given

# ******************** Creation ******************** #


@scenario('The Epsilon Greedy object is created')
def test_the_epsilon_greedy_object_is_created():
    return


@given('A created Epsilon Greedy object')
def a_created_epsilon_greedy_object(context, mocker):
    context.epsilon = random.random()
    context.arms = [mocker.Mock(), mocker.Mock(), mocker.Mock()]
    context.algorithm = EpsilonGreedy(epsilon=context.epsilon,
                                      arms=context.arms)
    context.expected = dict(epsilon=context.epsilon,
                            arms=context.arms)
    context.actual = {}
    return


@when('the arms are checked')
def the_arms_are_checked(context):
    context.actual["arms"] = context.algorithm.arms
    return


@And('the epsilon is checked')
def the_epsilon_is_checked(context):
    context.actual["epsilon"] = context.algorithm.epsilon
    return


@then('they have the expected values')
def they_have_the_expected_values(context):
    expect(context.actual).to(equal(context.expected))
    return

# ******************** best arm ******************** #


@scenario("The best-arm is retrieved")
def test_best_arm():
    return

#   Given A created Epsilon Greedy object


@and_given("the Epsilon Greedy object has been called")
def setup_epsilon_greedy_call_simulation(context):
    context.algorithm._rewards = numpy.array([5, 10, 2])
    context.expected = 1
    return


@when("the best-arm is retrieved")
def get_best_arm(context):
    context.best_arm = context.algorithm.best_arm
    return


@then("it is the arm with the most reward so far")
def check_best_arm(context):
    expect(context.best_arm).to(equal(context.expected))
    return

# ******************** counts ******************** #


@scenario("The counts are retrieved")
def test_counts():
    return

#  Given A created Epsilon Greedy object


@when("the counts are retrieved")
def get_counts(context):
    context.actual = context.algorithm.counts
    return


@then("they are an array of zeros")
def check_zeros(context):
    expect(all(context.actual == 0)).to(be_true)
    return


@and_also("they have the same length as the arms")
def check_length():
    return
