import pandas as pd
from pandas.testing import assert_frame_equal


def processing_fn(df):
    processed = df.assign(val3=df.val1 / df.val2)
    return processed


def test_processing_fn():
    # Create test data
    df = pd.DataFrame({"val1": [1, 1, -1, -2, 2], "val2": [1, 2, -2, -1, 2]})

    # Get result
    result = processing_fn(df)

    # Create expected output
    expected = df.copy()
    expected["val3"] = [1, 0.5, 0.5, 2, 1]

    # Test
    assert_frame_equal(result, expected, check_dtype=False)
