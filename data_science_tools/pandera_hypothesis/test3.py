import pandas as pd
import pandera as pa
import pytest
from pandas.testing import assert_frame_equal

expected = pa.DataFrameSchema(
    {
        "val1": pa.Column(int, pa.Check.in_range(-2, 3)),
        "val2": pa.Column(int, pa.Check.in_range(-2, 3)),
        "val3": pa.Column(float, pa.Check.in_range(-2, 3)),
    }
)


@pa.check_output(expected)
def processing_fn(df):
    processed = df.assign(val3=df.val1 / df.val2)
    return processed


val1 = [[1, 1, -1, -2, 2], [1, 1, -1, -2, 2]]
val2 = [[1, 2, -2, -1, 2], [1, 1, 1, 1, 1]]


@pytest.mark.parametrize("val1,val2", list(zip(val1, val2)))
def test_processing_fn(val1, val2):
    # Create test data
    df = pd.DataFrame({"val1": val1, "val2": val2})

    # Get result
    processing_fn(df)
