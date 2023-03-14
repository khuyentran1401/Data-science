Feature: Sentiment Analysis with Specific Word
  As a data scientist
  I want to ensure that the presence of a specific word has a positive or negative effect on the sentiment score of a text

  Scenario: Sentiment analysis with specific word
    Given a sentence 'I love this product' 
    And the same sentence with the addition of the word 'awesome'
    When I input the new sentence into the model
    Then the sentiment score should increase
