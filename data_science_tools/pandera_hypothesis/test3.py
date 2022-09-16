import pandas as pd
import pandera as pa

out_schema = pa.DataFrameSchema(
    {
        "val1": pa.Column(int, pa.Check.in_range(-2, 3)),
        "val2": pa.Column(int, pa.Check.in_range(-2, 3)),
        "val3": pa.Column(float, pa.Check.in_range(-2, 3)),
    }
)


@pa.check_output(out_schema)
def processing_fn(df):
    processed = df.assign(val3=df.val1 / df.val2)
    return processed


if __name__ == "__main__":
    df = pd.DataFrame({"val1": [1, 1, -1, -2, 2], "val2": [1, 1, -1, -2, 2]})
    processing_fn(df)
