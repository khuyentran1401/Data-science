Feature: Sentiment Analysis

  Scenario: Paraphrased text
    Given a text with a positive sentiment
    When the text is paraphrased
    Then the sentiment analysis result should remain positive