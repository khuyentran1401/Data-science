from textblob import TextBlob


def test_sentiment_score_increase():
    sent = "I love this product"
    word = "awesome"
    new_sent = " ".join([sent, word])

    # Get sentiment score
    sentiment_score = TextBlob(sent).sentiment.polarity
    adjusted_score = TextBlob(new_sent).sentiment.polarity

    # Check if the sentiment score increases
    assert adjusted_score > sentiment_score
