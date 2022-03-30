from textblob import TextBlob
import pytest


def extract_sentiment(text: str):
    """Extract sentiment using textblob.
    Polarity is within range [-1, 1]"""

    text = TextBlob(text)

    return text.sentiment.polarity


testdata = [
    "I think today will be a great day",
    "I do not think this will turn out well",
]


@pytest.mark.parametrize("sample", testdata)
def test_extract_sentiment(sample):

    sentiment = extract_sentiment(sample)

    assert sentiment > 0
