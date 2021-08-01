# Feature: Memory for the epsilon agent

# python
from functools import partial

import random

# pypi
from expects import (
    be_none,
    be_within,
    equal,
    expect,
)
from pytest_bdd import given, then, when

import pytest_bdd

scenario = partial(pytest_bdd.scenario, "../features/epsilon_memory.feature") 
And = given
and_also = then

# for testing
from .fixtures import katamari

# code being tested
from reinforcement_learning.bandit_algorithms.k_armed_bandit import AverageMemory


# ********** average memory creation

@scenario("The Average Memory is created")
def test_average_memory():
    return

@given("an average memory for the epsilon agent")
def setup_average_memory(katamari):
    katamari.arms = random.randint(1, 100)
    katamari.memory = AverageMemory(arms=katamari.arms)
    return

@when("the average memory is checked")
def check_average_memory(katamari):
    return

@then("it has the right expected reward")
def check_expected_reward(katamari):
    expect(sum(katamari.memory.expected_reward)).to(equal(0))
    expect(len(katamari.memory.expected_reward)).to(equal(katamari.arms))
    return

@and_also("it has the right number of pulled arms")
def check_pulled_arms(katamari):
    expect(len(katamari.memory.pulled)).to(equal(katamari.arms))
    expect(sum(katamari.memory.expected_reward)).to(equal(0))
    return

# ********** Best Arm

@scenario("The best arm is retrieved")
def test_best_arm():
    return

#   Given an average memory for the epsilon agent

@when("the best arm is retrieved")
def get_best_arm(katamari):
    katamari.memory._expected_reward = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    katamari.expected = 5
    katamari.actual = katamari.memory.best_arm
    return


@then("it is the best arm")
def check_best_arm(katamari):
    expect(katamari.actual).to(equal(katamari.expected))
    return

# ********** arm update

@scenario("An arm is updated")
def test_arm_update():
    return

#   Given an average memory for the epsilon agent

@when("the arm's reward is updated")
def update_arm(katamari):
    katamari.last_arm = random.randint(1, katamari.arms)
    katamari.expected_reward = 1
    katamari.memory.update(katamari.last_arm, 1)
    return


@then("the pulled count is updated")
def check_pulled_count(katamari):
    katamari.actual_count = katamari.memory.pulled[katamari.last_arm]
    return


@and_also("the expected reward is updated")
def check_updated_expected_reward(katamari):
    katamari.actual_reward = katamari.memory.expected_reward[
        katamari.last_arm]
    expect(katamari.actual_reward).to(
        equal(katamari.expected_reward))
    return
