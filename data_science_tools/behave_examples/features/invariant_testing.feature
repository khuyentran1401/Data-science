Feature: Sentiment Analysis
  As a data scientist
  I want to ensure that my model is invariant to paraphrasing 
  So that my model can produce consistent results in real-world scenarios.

  Scenario: Paraphrased text
    Given a text
    When the text is paraphrased
    Then both text should have the same sentiment