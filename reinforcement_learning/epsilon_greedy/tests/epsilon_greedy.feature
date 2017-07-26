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
