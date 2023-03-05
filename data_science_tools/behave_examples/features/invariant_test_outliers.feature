Feature: Model Performance

  Scenario: Outliers
    Given a original dataset without outliers exists
    And the performance when testing on the original dataset exists
    And a dataset with outliers exists
    When the model is trained on the dataset with outliers
    Then the model's performance should not be significantly affected