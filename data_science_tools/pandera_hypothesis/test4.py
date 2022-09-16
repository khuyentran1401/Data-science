import hypothesis
import pandera as pa

schema = pa.DataFrameSchema(
    {
        "val1": pa.Column(int, pa.Check.in_range(-2, 3)),
        "val2": pa.Column(int, pa.Check.in_range(-2, 3)),
    }
)

out_schema = schema.add_columns(
    {
        "val3": pa.Column(float, pa.Check.in_range(-2, 3)),
    },
)


@pa.check_output(out_schema)
def processing_fn(df):
    processed = df.assign(val3=df.val1 / df.val2)
    return processed


@hypothesis.given(schema.strategy(size=5))
def test_processing_fn(dataframe):
    processing_fn(dataframe)
