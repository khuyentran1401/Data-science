Feature: Customer Churn Prediction

  As a business analyst
  I want to predict whether a customer will churn or not
  So that I can take appropriate actions to retain customers

  Scenario: Predict customer churn
    Given some features about a customer 
    And churn prediction model "churn_model.pkl" exists
    When I run the churn prediction model
    Then the predicted churn status should be either "No" or "Yes"
