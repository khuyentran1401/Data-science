Feature: Sentiment Analysis
  As a data scientist
  I want to ensure that my model is invariant to paraphrasing 
  So that my model can produce consistent results in real-world scenarios.

  Scenario: Paraphrased text
    Given a text with a positive sentiment
    When the text is paraphrased
    Then the sentiment analysis result should remain positive