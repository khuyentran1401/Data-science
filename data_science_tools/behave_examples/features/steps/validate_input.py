import pandas as pd
from behave import given, then, when


@given("the value of marital_status in at least one sample is incorrect")
def step_impl(context):
    # Create a new pandas DataFrame with the extracted data
    context.customer_data = pd.DataFrame(
        context.table.rows, columns=context.table.headings
    )


@when("the input is validated")
def step_impl(context):
    if not context.customer_data.marital_status.isin(["Married", "Single"]).all():
        context.message = "The value for marital_status is incorrect."
    else:
        context.message = None


@then("an error message should be returned")
def step_impl(context):
    assert context.message == "The value for marital_status is incorrect."