Feature: Memory for the epsilon agent

Scenario: The Average Memory is created
  Given an average memory for the epsilon agent
  When the average memory is checked
  Then it has the right expected reward
  And it has the right number of pulled arms

Scenario: The best arm is retrieved
  Given an average memory for the epsilon agent
  When the best arm is retrieved
  Then it is the best arm

Scenario: An arm is updated
  Given an average memory for the epsilon agent
  When the arm's reward is updated
  Then the pulled count is updated
  And the expected reward is updated
