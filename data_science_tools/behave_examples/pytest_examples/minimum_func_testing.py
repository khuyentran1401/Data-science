from typing import Union

import numpy as np
import pytest
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

    # Predict the output using the input array
    output = model.predict(input_array)

    return output


@pytest.mark.parametrize(
    "input_value,expected_size", [(42, 1), (3.14, 1), ([1, 2, 3], 3)]
)
def test_predict(input_value, expected_size):
    output = predict(input_value)
    assert isinstance(output, np.ndarray)
    assert all(isinstance(x, (int, float)) for x in output)
    assert len(output) == expected_size
