import pytest
from textblob import TextBlob


def extract_sentiment(text: str):
        '''Extract sentiment using textblob. 
        Polarity is within range [-1, 1]'''

        text = TextBlob(text)

        return text.sentiment.polarity

def text_contain_word(word: str, text: str):
    '''Find whether the text contains a particular word'''
    
    return word in text

@pytest.fixture
def example_data():
    return 'Today I found a duck and I am happy'

def test_extract_sentiment(example_data):

    sentiment = extract_sentiment(example_data)

    assert sentiment > 0

def test_text_contain_word(example_data):

    word = 'duck'

    assert text_contain_word(word, example_data) == True
