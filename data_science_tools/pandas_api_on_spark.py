# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==1.26.4",
#     "pandas==2.2.3",
#     "plotly==6.0.1",
#     "pyarrow==20.0.0",
#     "pyspark==3.5.5",
#     "scikit-learn==1.6.1",
#     "narwhals==1.36.0",
# ]
# ///

import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Motivation""")
    return


@app.cell(hide_code=True)
def _():
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    return


@app.cell
def _():
    import pandas as pd

    pandas_df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    print(pandas_df["value"].mean())
    return pandas_df, pd


@app.cell
def _():
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import avg

    spark = SparkSession.builder.getOrCreate()

    spark_df = spark.createDataFrame([(1,), (2,), (3,), (4,), (5,)], ["value"])
    spark_df.select(avg("value")).show()
    return (spark,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Basic Operations""")
    return


@app.cell
def _():
    import pyspark.pandas as ps

    ps_s = ps.Series([1, 3, 5, 6, 8])
    return (ps,)


@app.cell
def _(pd):
    import numpy as np

    ps_df = pd.DataFrame(
        {"id": np.arange(1, 1_000_001), "value": np.random.randn(1_000_000)}
    )
    return (ps_df,)


@app.cell
def _(pandas_df, ps):
    ps_df_from_pandas = ps.from_pandas(pandas_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Basic Operations""")
    return


@app.cell
def _(ps_df):
    ps_df.describe()
    return


@app.cell
def _(ps_df):
    # Display the summary of the DataFrame
    ps_df.info()

    return


@app.cell
def _(ps_df):
    ps_df.head()
    return


@app.cell
def _(ps_df):
    # Filter rows and drop any NaN values
    filtered_df = ps_df.where(ps_df.value > 0).dropna()
    filtered_df.head()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## GroupBy""")
    return


@app.cell
def _(ps):
    ps_df_2 = ps.DataFrame(
        {"category": ["A", "B", "A", "C", "B"], "value": [10, 20, 15, 30, 25]}
    )
    return (ps_df_2,)


@app.cell
def _(ps_df_2):
    ps_df_2.groupby("category").value.mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Plotting""")
    return


@app.cell
def _(ps_df):
    ps_df["value"].plot.hist()
    return


@app.cell
def _(ps_df_2):
    ps_df_2.plot.bar(x="category", y="value")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Reading And Writing Data""")
    return


@app.cell
def _(ps, ps_df):
    ps_df.to_csv("output_data.csv", index=False)
    ps.read_csv("output_data.csv").head()
    return


@app.cell
def _(ps, ps_df):
    ps_df.to_parquet("output_data.parquet")
    ps.read_parquet("output_data.parquet").head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Using Pandas API on Spark in Conjunction with Regular Pandas""")
    return


@app.cell
def _(ps):
    from sklearn.linear_model import LinearRegression

    # Create a large Pandas API on Spark DataFrame
    large_pdf_df = ps.DataFrame(
        {
            "feature1": range(1_000_000),
            "feature2": range(1_000_000, 2_000_000),
            "target": range(500_000, 1_500_000),
        }
    )
    print(f"Length of the original DataFrame: {len(large_pdf_df):,}")

    # Aggregate the data to a smaller size
    aggregated = large_pdf_df.groupby(large_pdf_df.feature1 // 10000).mean()
    print(f"Length of the aggregated DataFrame: {len(aggregated):,}")

    # Convert to pandas DataFrame
    small_pdf = aggregated.to_pandas()

    # Train a scikit-learn model
    model = LinearRegression()
    X = small_pdf[["feature1", "feature2"]]
    y = small_pdf["target"]
    model.fit(X, y)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Pandas API on Spark Query Execution Model""")
    return


@app.cell
def _(pandas_df):
    pandas_df["value"] = pandas_df["value"] + 1  # Operation executes immediately
    print(pandas_df)
    return


@app.cell
def _(ps_df):
    # Using Pandas API on Spark
    updated_psdf = ps_df.assign(a=ps_df["value"] + 1)  # Lazy operation
    print(updated_psdf.head())  # Triggers actual computation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Pandas API on Spark vs PySpark""")
    return


@app.cell
def _(spark):
    from pyspark.sql.functions import col

    pyspark_df = spark.createDataFrame([(1, 4), (2, 5), (3, 6)], ["col1", "col2"])
    pyspark_df.select((col("col1") + col("col2")).alias("sum")).show()
    return (col,)


@app.cell
def _(ps):
    pandas_spark_df = ps.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    (pandas_spark_df["col1"] + pandas_spark_df["col2"]).head()
    return (pandas_spark_df,)


@app.cell
def _(col, pandas_spark_df):
    # Convert Pandas API on Spark DataFrame to PySpark DataFrame
    spark_native_df = pandas_spark_df.to_spark()

    # Now you can use full PySpark functionality
    spark_native_df.select((col("col1") + col("col2")).alias("sum")).show()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
