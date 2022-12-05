import hypothesis
import pandas as pd
import pandera as pa
import pytest

schema = pa.DataFrameSchema(
    {
        "val1": pa.Column(int, pa.Check.in_range(-2, 3)),
        "val2": pa.Column(int, pa.Check.in_range(-2, 3)),
    }
)

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


@hypothesis.given(schema.strategy(size=5))
def test_processing_fn(df):
    # Get result
    processing_fn(df)
