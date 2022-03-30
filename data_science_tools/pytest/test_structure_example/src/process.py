from textblob import TextBlob


def extract_sentiment(text: str):
    """Extract sentiment using textblob.
    Polarity is within range [-1, 1]"""

    text = TextBlob(text)

    return text.sentiment.polarity


def text_contain_word(word: str, text: str):
    """Find whether the text contains a particular word"""

    return word in text


testdata = [("There is a duck in this text", True), ("There is nothing here", False)]
