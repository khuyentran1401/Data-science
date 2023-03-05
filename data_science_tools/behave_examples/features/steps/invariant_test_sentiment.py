from behave import given, then, when
from textblob import TextBlob


@given("a text with a positive sentiment")
def step_given_positive_sentiment(context):
    context.sent = "This is a great day!"


@when("the text is paraphrased")
def step_when_paraphrased(context):
    context.sent_paraphrased = "Today is fantastic!"


@then("the sentiment analysis result should remain positive")
def step_then_sentiment_analysis(context):
    sentiment_original = TextBlob(context.sent).sentiment.polarity
    sentiment_paraphrased = TextBlob(context.sent_paraphrased).sentiment.polarity
    assert sentiment_original == sentiment_paraphrased
