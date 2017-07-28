# coding=utf-8
"""Epsilon Greedy Implementation feature tests."""
# python standard library
from functools import partial
import random

# pypi
from expects import (
    be,
    be_none,
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
and_when = when
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


@and_when('the epsilon is checked')
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

# ******************** rewards ******************** #


@scenario("The initial rewards are retrieved")
def test_initial_rewards():
    return

#   Given A created Epsilon Greedy object


@when("the initial rewards are retrieved")
def get_rewards(context):
    context.actual = context.algorithm.rewards
    return

#   Then they are an array of zeros
#   and_when they have the same length as the arms


# ******************** reset ******************** #


@scenario("The reset method is called")
def test_reset():
    return

#   Given A created Epsilon Greedy object


@when("the rewards and counts are set")
def set_rewards_and_counts(context):
    expect(context.algorithm.rewards).to_not(be_none)
    expect(context.algorithm.counts).to_not(be_none)
    return


@and_when("the reset method is called")
def call_reset(context):
    context.algorithm.reset()
    return


@then("the rewards and counts unset")
def check_reset(context):
    expect(context.algorithm._counts).to(be_none)
    expect(context.algorithm._rewards).to(be_none)
    return

# ******************** call ******************** #


@scenario("The call method is called")
def test_call():
    return

#   Given A created Epsilon Greedy object


@when("the Epsilon Greedy object is called")
def call_epsilon_greedy(context, mocker):
    context.select_arm = mocker.MagicMock()
    context.update = mocker.MagicMock()
    context.arm = random.randrange(len(context.arms))
    context.algorithm.select_arm = context.select_arm
    context.algorithm.update = context.update
    context.algorithm.select_arm.return_value = context.arm
    context.algorithm()
    return


@then("the select_arm method is called")
def check_select_arm(context):
    context.select_arm.assert_called_once_with()
    return


@and_also("the update_rewards method is called with the arm given")
def check_update_rewards(context):
    context.update.assert_called_once_with(context.arm)
    return

# ******************** select_arm ******************** #
# ********** exploitation ********** #


@scenario("The select_arm method is called and exploitation is called for")
def test_exploitation():
    return

#   Given A created Epsilon Greedy object


@when("the random value is greater than epsilon")
def set_exploitation_value(context, mocker):
    context.random = mocker.MagicMock(spec=random)
    context.random.random.return_value = context.epsilon + 0.1 
    mocker.patch("{0}.random".format(EpsilonGreedy.__module__), context.random)
    return


@and_when("the select_arm method is called")
def call_select_arm(context):
    context.algorithm._rewards = numpy.array([1, 2, 3])
    context.selected = context.algorithm.select_arm()
    return


@then("the best arm's index is returned")
def expect_best_arm(context):
    expect(context.selected).to(equal(context.algorithm.best_arm))
    return

# ********** exploration ********** #


@scenario("The select_arm method is called and exploration is called for")
def test_exploration():
    return

#   Given A created Epsilon Greedy object


@when("the random value is less than epsilon")
def set_exploration_value(context, mocker):
    context.random = mocker.MagicMock(spec=random)
    context.random.random.return_value = context.epsilon - 0.05
    context.expected = 1
    context.random.randrange.return_value = context.expected
    mocker.patch("{0}.random".format(EpsilonGreedy.__module__), context.random)
    return

#   And the select_arm method is called


@then("a random arm's index is returned")
def check_random_arm(context):
    expect(context.selected).to(equal(context.expected))
    context.random.randrange.assert_called_once_with(len(context.arms))
    return

# ******************** update ******************** #


@scenario("The update method is called")
def test_update():
    return

#   Given A created Epsilon Greedy object


@when("the update method is called")
def call_update(context):
|     context.arm = random.randrange(len(context.arms))
|     return


| @then("the count is updated")
| def check_count():
|     return


@and_also("the reward is updated")
def check_reward():
    return
