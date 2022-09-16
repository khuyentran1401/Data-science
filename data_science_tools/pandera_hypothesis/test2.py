import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


def processing_fn(df):
    processed = df.assign(val3=df.val1 / df.val2)
    return processed


val1 = [[1, 1, -1, -2, 2], [1, 1, -1, -2, 2]]
val2 = [[1, 2, -2, -1, 2], [1, 1, 1, 1, 1]]
val3 = [[1, 0.5, 0.5, 2, 1], [1, 1, -1, -2, 2]]


@pytest.mark.parametrize("val1,val2,val3", list(zip(val1, val2, val3)))
def test_processing_fn(val1, val2, val3):
    # Create test data
    df = pd.DataFrame({"val1": val1, "val2": val2})

    # Get result
    result = processing_fn(df)

    # Create expected output
    expected = df.copy()
    expected["val3"] = val3

    # Test
    assert_frame_equal(result, expected, check_dtype=False)
