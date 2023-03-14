Feature: Test my_ml_model

  Scenario: Test integer input
    Given I have an integer input of 42
    When I run the model
    Then the output should be an array of one number 

  Scenario: Test float input
    Given I have a float input of 3.14
    When I run the model
    Then the output should be an array of one number

  Scenario: Test list input
    Given I have a list input of [1, 2, 3]
    When I run the model
    Then the output should be an array of three numbers