import joblib
import pandas as pd
from behave import given, when, then
from train_customer_churn import preprocess_data


@given("some features about a customer")
def step_impl(context):
    context.customer_data = pd.DataFrame(
        {
            "customer_id": [32],
            "gender": ["Male"],
            "age": [55],
            "marital_status": ["Married"],
            "education": ["High School"],
            "occupation": ["Blue Collar"],
            "income": [80000],
            "account_balance": [7000],
            "credit_score": [700],
        }
    )


@given('churn prediction model "churn_model.pkl" exists')
def step_impl(context):
    context.model = joblib.load("churn_model.pkl")


@when("I run the churn prediction model")
def step_impl(context):
    model = context.model
    processed_data = preprocess_data(context.customer_data)
    context.prediction = model.predict(processed_data)


@then('the predicted churn status should be either "No" or "Yes"')
def step_impl(context):
    assert context.prediction in ["No", "Yes"]
