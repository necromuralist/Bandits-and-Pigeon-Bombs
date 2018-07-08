# coding=utf-8
"""Bernoulli Arm feature tests."""
# python standard library
from functools import partial
import random

# pypi
from expects import (
    equal,
    expect,
)
from pytest_bdd import (
    given,
    then,
    when,
)
import pytest_bdd

# this project
from ..bernoulli_arm import BernoulliArm
from .fixtures import context  # noqa: F401

scenario = partial(pytest_bdd.scenario, "bernoulli_arm.feature")


# ******************** The class exists ******************** #
@scenario('A Bernoulli Arm is created')
def test_a_bernoulli_arm_is_created():
    return


@given('A created Bernoulli Arm object')  # noqa: F811
def a_created_bernoulli_arm_object(context):
    context.expected_probability = random.random()
    context.expected_reward = random.randrange(10)
    context.expected_penalty = random.randrange(10)
    context.arm = BernoulliArm(context.expected_probability,
                               context.expected_reward,
                               context.expected_penalty)
    return


@when('the properties are checked')  # noqa: F811
def the_properties_are_checked(context):
    context.actual_probability = context.arm.probability
    context.actual_reward = context.arm.reward
    context.actual_penalty = context.arm.penalty
    return


@then('it has the correct properties')  # noqa: F811
def it_has_the_correct_properties(context):
    expect(context.actual_probability).to(equal(context.expected_probability))
    expect(context.actual_reward).to(equal(context.expected_reward))
    expect(context.actual_penalty).to(equal(context.expected_penalty))
    return

# ******************** pull the arm ******************** #
# ********** the player loses ********** #


@scenario("The arm is pulled and the player loses")
def test_losing():
    return

#  Given A created Bernoulli Arm object


@when("the player pulls the arm and loses")  # noqa: F811
def losing_pull(context, mocker):
    random_mock = mocker.MagicMock(spec=random)
    random_mock.random.return_value = context.expected_probability + 0.1
    mocker.patch("{}.random".format(BernoulliArm.__module__), random_mock)
    context.outcome = context.arm()
    return


@then("the player receives the penalty")  # noqa: F811
def check_penalty(context):
    expect(context.outcome).to(equal(context.expected_penalty))
    return

# ********** the player wins ********** #


@scenario("The arm is pulled and the player wins")
def test_win():
    return

#  Given A created Bernoulli Arm object


@when("the player pulls the arm and wins")  # noqa: F811
def winning_pull(context, mocker):
    random_mock = mocker.MagicMock()
    random_mock.random.return_value = context.expected_probability - 0.001
    mocker.patch("{}.random".format(BernoulliArm.__module__), random_mock)
    context.outcome = context.arm()
    return


@then("the player receives the reward")  # noqa: F811
def check_reward(context):
    expect(context.outcome).to(equal(context.expected_reward))
    return
