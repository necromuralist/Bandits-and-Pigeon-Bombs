# python
from functools import partial

import random

# pypi
from expects import (
    be_within,
    equal,
    expect,
)
from pytest_bdd import given, then, when

import pytest_bdd

scenario = partial(pytest_bdd.scenario, "../features/bandit_arm.feature") 
And = given

# for testing
from .fixtures import katamari

# code being tested
from reinforcement_learning.bandit_algorithms.k_armed_bandit import Bandit

# ******** Pull an arm
@scenario("Pulling an arm.")
def test_arm_pull():
    return

@given("I'm a player")
def setup_bandit_player(katamari):
    katamari.k = random.randrange(1, 100)
    katamari.bandit = Bandit(k=katamari.k)
    return

@And("I pull a valid arm")
def pull_arm(katamari):
    expect(len(katamari.bandit.arms)).to(equal(katamari.k))
    katamari.arm = random.randrange(katamari.bandit.k)
    return

@when("I check the outcome")
def check_outcome(katamari):
    katamari.outcome = katamari.bandit(katamari.arm)
    return

@then("the reward is an expected value")
def expect_value(katamari):
    center = katamari.bandit.arms[katamari.arm].center
    # expect(katamari.outcome).to(be_within(center - 1, center + 1))
    return

# ********** Check the best arm

@scenario("Checking the best arm.")
def test_best_arm():
    return

#    Given I'm a player

@when("I check the bandit's best arm.")
def check_best(katamari):
    highest = max((arm.center for arm in katamari.bandit.arms)) + 100
    katamari.expected = random.randrange(katamari.k)
    katamari.bandit.arms[katamari.expected]._center = highest
    katamari.actual = katamari.bandit.best_arm
    return


@then("it's the arm with the highest center.")
def expected_highest_arm(katamari):
    expect(katamari.actual).to(equal(katamari.expected))
    return
