Feature: Epsilon Greedy Implementation

Scenario: The Epsilon Greedy object is created
  Given A created Epsilon Greedy object
  When the epsilon is checked
  And the arms are checked
  Then they have the expected values

Scenario: The best-arm is retrieved
  Given A created Epsilon Greedy object
  And the Epsilon Greedy object has been called
  When the best-arm is retrieved
  Then it is the arm with the most reward so far

Scenario: The counts are retrieved
  Given A created Epsilon Greedy object
  When the counts are retrieved
  Then they are an array of zeros
  And they have the same length as the arms

Scenario: The initial rewards are retrieved
  Given A created Epsilon Greedy object
  When the initial rewards are retrieved
  Then they are an array of zeros
  And they have the same length as the arms

Scenario: The reset method is called
  Given A created Epsilon Greedy object
  When the rewards and counts are set
  And the reset method is called
  Then the rewards and counts unset
  
Scenario: The call method is called
  Given A created Epsilon Greedy object
  When the Epsilon Greedy object is called
  Then the select_arm method is called
  And the update_rewards method is called with the arm given

Scenario: The select_arm method is called and exploitation is called for
  Given A created Epsilon Greedy object
  When the random value is greater than epsilon
  And the select_arm method is called
  Then the best arm's index is returned

Scenario: The select_arm method is called and exploration is called for
  Given A created Epsilon Greedy object
  When the random value is less than epsilon
  And the select_arm method is called
  Then a random arm's index is returned

Scenario: The update method is called
  Given A created Epsilon Greedy object
  When the update method is called
  Then the count is updated
  And the reward is updated
