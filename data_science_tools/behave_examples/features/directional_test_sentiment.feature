Feature: Sentiment Analysis
  As a user of the sentiment analysis model
  I want to ensure that the model predicts the correct sentiment
  So that I can trust the model's predictions

  Scenario: Positive sentiment prediction
    Given a positive text
    When I input the text into the model
    Then the model should predict "positive" sentiment
