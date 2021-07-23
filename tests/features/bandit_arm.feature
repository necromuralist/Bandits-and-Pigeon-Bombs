Feature: k-Armed Bandit
  A multi-armed bandit to play.

  Scenario: Pulling an arm.
    Given I'm a player
    And I pull a valid arm
    When I check the outcome
    Then the reward is an expected value

  Scenario: Checking the best arm.
    Given I'm a player
    When I check the bandit's best arm.
    Then it's the arm with the highest center.
