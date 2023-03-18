from typing import Union

import numpy as np
from behave import given, then, when
from sklearn.linear_model import LinearRegression


def predict(input_data: Union[int, float, str, list]):
    # Reshape the input data
    if isinstance(input_data, (int, float, list)):
        input_array = np.array(input_data).reshape(-1, 1)
    else:
        raise ValueError("Input type not supported")

    # Create a linear regression model
    model = LinearRegression()

    # Train the model on a sample dataset
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    model.fit(X, y)

    return model.predict(input_array)


@given("I have an integer input of {input_value}")
def step_given_integer_input(context, input_value):
    context.input_value = int(input_value)


@given("I have a float input of {input_value}")
def step_given_float_input(context, input_value):
    context.input_value = float(input_value)


@given("I have a list input of {input_value}")
def step_given_list_input(context, input_value):
    context.input_value = eval(input_value)


@when("I run the model")
def step_when_run_model(context):
    context.output = predict(context.input_value)


@then("the output should be an array of one number")
def step_then_check_output_num(context):
    assert isinstance(context.output, np.ndarray)
    assert all(isinstance(x, (int, float)) for x in context.output)
    assert len(context.output) == 1


@then("the output should be an array of three numbers")
def step_then_check_output_list(context):
    assert isinstance(context.output, np.ndarray)
    assert all(isinstance(x, (int, float)) for x in context.output)
    assert len(context.output) == 3
