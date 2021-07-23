Feature: EpsilonExplorer
  An epsilon-greedy explorer.

 Scenario: Preparing to run the explorer
   Given an Epsilon Explorer
   When the properties are checked
   Then they have the right length
   And they are all zero
