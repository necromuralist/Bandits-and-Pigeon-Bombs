Feature: Bernoulli Arm

Scenario: A Bernoulli Arm is created
  Given A created Bernoulli Arm object
  When the properties are checked
  Then it has the correct properties

Scenario: The arm is pulled and the player loses
  Given A created Bernoulli Arm object
  When the player pulls the arm and loses
  Then the player receives the penalty

Scenario: The arm is pulled and the player wins
  Given A created Bernoulli Arm object
  When the player pulls the arm and wins
  Then the player receives the reward
