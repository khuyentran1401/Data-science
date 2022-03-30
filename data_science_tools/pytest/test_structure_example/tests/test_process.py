from src.process import extract_sentiment, text_contain_word
import pytest


@pytest.fixture
def example_data():
    return "Today I found a duck and I am happy"


def test_extract_sentiment():

    text = "Today I found a duck and I am happy"

    sentiment = extract_sentiment(text)

    assert sentiment > 0


def test_text_contain_word(example_data):

    word = "duck"

    assert text_contain_word(word, example_data) == True
