from behave import given, then, when
from textblob import TextBlob


@given("a text")
def step_given_positive_sentiment(context):
    context.sent = "The hotel room was great! It was spacious, clean and had a nice view of the city."


@when("the text is paraphrased")
def step_when_paraphrased(context):
    context.sent_paraphrased = "The hotel room wasn't bad. It wasn't cramped, dirty, and had a decent view of the city."


@then("both text should have the same sentiment")
def step_then_sentiment_analysis(context):
    # Get sentiment of each sentence
    sentiment_original = TextBlob(context.sent).sentiment.polarity
    sentiment_paraphrased = TextBlob(context.sent_paraphrased).sentiment.polarity

    # Print sentiment
    print(f"Sentiment of the original text: {sentiment_original:.2f}")
    print(f"Sentiment of the paraphrased sentence: {sentiment_paraphrased:.2f}")

    # Assert that both sentences have the same sentiment
    both_positive = (sentiment_original > 0) and (sentiment_paraphrased > 0)
    both_negative = (sentiment_original < 0) and (sentiment_paraphrased < 0)
    assert both_positive or both_negative
