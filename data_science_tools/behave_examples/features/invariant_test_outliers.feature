Feature: Model Performance
  As a data scientist
  I want to guarantee my model is invariant to variations in the input data 
  So that my model can produce accurate forecasts in real-world scenarios.

  Scenario: Outliers
    Given a original dataset without outliers exists
    And the performance when testing on the original dataset exists
    And a dataset with outliers exists
    When the model is trained on the dataset with outliers
    Then the model's performance should not be significantly affected