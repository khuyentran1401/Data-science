from textblob import TextBlob


def test_sentiment_the_same_after_paraphrasing():
    sent = "The hotel room was great! It was spacious, clean and had a nice view of the city."
    sent_paraphrased = "The hotel room wasn't bad. It wasn't cramped, dirty, and had a decent view of the city."
    sentiment_original = TextBlob(sent).sentiment.polarity
    sentiment_paraphrased = TextBlob(sent_paraphrased).sentiment.polarity
    both_positive = (sentiment_original > 0) and (sentiment_paraphrased > 0)
    both_negative = (sentiment_original < 0) and (sentiment_paraphrased < 0)
    assert both_positive or both_negative
