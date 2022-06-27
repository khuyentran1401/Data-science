from textblob import TextBlob


def extract_sentiment(text: str):
    """Extract sentiment using textblob.
    Polarity is within range [-1, 1]"""

    text = TextBlob(text) 

    return text.sentiment.polarity 


def test_extract_sentiment(): 

    text = "I think today will be a great day"

    sentiment = extract_sentiment(text)

    assert sentiment > 0


def test_extract_sentiment_negative():

    text = "I do not think this will turn out well"

    sentiment = extract_sentiment(text)

    assert sentiment < 0