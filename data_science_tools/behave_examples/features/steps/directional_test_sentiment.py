from behave import given, then, when
from textblob import TextBlob


@given("a positive text")
def step_given_positive_text(context):
    context.sent = "I love this product"


@when("I input the text into the model")
def step_when_use_model(context):
    polarity = TextBlob(context.sent).sentiment.polarity
    if polarity > 0:
        context.sentiment = "positive"
    else:
        context.sentiment = "negative"


@then('the model should predict "positive" sentiment')
def step_then_positive(context):
    assert context.sentiment == "positive"
