from behave import given, then, when
from textblob import TextBlob


@given("a sentence")
def step_given_positive_word(context):
    context.sent = "I love this product"


@given("the same sentence with the addition of the word '{word}'")
def step_given_a_positive_word(context, word):
    context.new_sent = f"I love this {word} product"


@when("I input the new sentence into the model")
def step_when_use_model(context):
    context.sentiment_score = TextBlob(context.sent).sentiment.polarity
    context.adjusted_score = TextBlob(context.new_sent).sentiment.polarity


@then("the sentiment score should increase")
def step_then_positive(context):
    assert context.adjusted_score > context.sentiment_score
