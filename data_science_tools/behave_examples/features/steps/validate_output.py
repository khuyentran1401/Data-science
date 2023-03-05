import joblib
import pandas as pd
from behave import given, then, when
from training import preprocess_data


@given("some features about a customer")
def step_impl(context):
    # Create a new pandas DataFrame with the extracted data
    context.customer_data = pd.DataFrame(
        context.table.rows, columns=context.table.headings
    )


@given('churn prediction model "churn_model.pkl" exists')
def step_impl(context):
    context.model = joblib.load("churn_model.pkl")


@when("I run the churn prediction model")
def step_impl(context):
    model = context.model
    processed_data = preprocess_data(context.customer_data)
    context.prediction = model.predict(processed_data)


@then('an error message should be returned  ')
def step_impl(context):
    assert context.prediction in ["No", "Yes"]
