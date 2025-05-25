# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "duckdb==1.2.2",
#     "marimo",
#     "narwhals==1.39.0",
#     "pandas==2.2.3",
#     "polars==1.29.0",
#     "pyarrow==20.0.0",
#     "pyspark==3.5.5",
#     "sqlframe==3.32.1",
# ]
# ///

import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Motivation""")
    return


@app.cell
def _():
    from datetime import datetime

    import pandas as pd

    df = pd.DataFrame(
        {
            "date": [datetime(2020, 1, 1), datetime(2020, 1, 8), datetime(2020, 2, 3)],
            "price": [1, 4, 3],
        }
    )
    df
    return datetime, df, pd


@app.cell
def _(df):
    def monthly_aggregate_pandas(user_df):
        return user_df.resample("MS", on="date")[["price"]].mean()

    monthly_aggregate_pandas(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Dataframe-agnostic data science

    Let's define a dataframe-agnostic function to calculate monthly average prices. It needs to support pandas, Polars, PySpark, DuckDB, PyArrow, Dask, and cuDF, without doing any conversion between libraries.

    ## Bad solution: just convert to pandas

    This kind of works, but:

    - It doesn't return to the user the same class they started with.
    - It kills lazy execution.
    - It kills GPU acceleration.
    - If forces pandas as a required dependency.
    """
    )
    return


@app.cell
def _():
    import duckdb
    import polars as pl
    import pyarrow as pa
    import pyspark
    import pyspark.sql.functions as F
    from pyspark.sql import SparkSession

    return F, SparkSession, duckdb, pa, pl, pyspark


@app.cell
def _(duckdb, pa, pd, pl, pyspark):
    def monthly_aggregate_bad(user_df):
        if isinstance(user_df, pd.DataFrame):
            df = user_df
        elif isinstance(user_df, pl.DataFrame):
            df = user_df.to_pandas()
        elif isinstance(user_df, duckdb.DuckDBPyRelation):
            df = user_df.df()
        elif isinstance(user_df, pa.Table):
            df = user_df.to_pandas()
        elif isinstance(user_df, pyspark.sql.dataframe.DataFrame):
            df = user_df.toPandas()
        else:
            raise TypeError("Unsupported DataFrame type: cannot convert to pandas")

        return df.resample("MS", on="date")[["price"]].mean()

    return (monthly_aggregate_bad,)


@app.cell
def _(datetime):
    data = {
        "date": [datetime(2020, 1, 1), datetime(2020, 1, 8), datetime(2020, 2, 3)],
        "price": [1, 4, 3],
    }
    return (data,)


@app.cell
def _(SparkSession, data, duckdb, monthly_aggregate_bad, pa, pd, pl):
    # pandas
    pandas_df = pd.DataFrame(data)
    monthly_aggregate_bad(pandas_df)

    # polars
    polars_df = pl.DataFrame(data)
    monthly_aggregate_bad(polars_df)

    # duckdb
    duckdb_df = duckdb.from_df(pandas_df)
    monthly_aggregate_bad(duckdb_df)

    # pyspark
    spark = SparkSession.builder.getOrCreate()
    spark_df = spark.createDataFrame(pandas_df)
    monthly_aggregate_bad(spark_df)

    # pyarrow
    arrow_table = pa.table(data)
    monthly_aggregate_bad(arrow_table)
    return arrow_table, duckdb_df, pandas_df, polars_df, spark_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Unmaintainable solution: different branches for each library

    This works, but is unfeasibly difficult to test and maintain, especially when also factoring in API changes between different versions of the same library (e.g. pandas `1.*` vs pandas `2.*`).
    """
    )
    return


@app.cell
def _(F, pd, pl, pyspark):
    def monthly_aggregate_unmaintainable(user_df):
        if isinstance(user_df, pd.DataFrame):
            result = user_df.resample("MS", on="date")[["price"]].mean()
        elif isinstance(user_df, pl.DataFrame):
            result = (
                user_df.group_by(pl.col("date").dt.truncate("1mo"))
                .agg(pl.col("price").mean())
                .sort("date")
            )
        elif isinstance(user_df, pyspark.sql.dataframe.DataFrame):
            result = (
                user_df.withColumn("date_month", F.date_trunc("month", F.col("date")))
                .groupBy("date_month")
                .agg(F.mean("price").alias("price_mean"))
                .orderBy("date_month")
            )
        # TODO: more branches for DuckDB, PyArrow, Dask, etc... :sob:
        return result

    return (monthly_aggregate_unmaintainable,)


@app.cell
def _(monthly_aggregate_unmaintainable, pandas_df, polars_df, spark_df):
    # pandas
    monthly_aggregate_unmaintainable(pandas_df)

    # polars
    monthly_aggregate_unmaintainable(polars_df)

    # pyspark
    monthly_aggregate_unmaintainable(spark_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Best solution: Narwhals as a unified dataframe interface

    - Preserves lazy execution and GPU acceleration.
    - Users get back what they started with.
    - Easy to write and maintain.
    - Strong and complete static typing.
    """
    )
    return


@app.cell
def _():
    import narwhals as nw
    from narwhals.typing import IntoFrameT

    def monthly_aggregate(user_df: IntoFrameT) -> IntoFrameT:
        return (
            nw.from_native(user_df)
            .group_by(nw.col("date").dt.truncate("1mo"))
            .agg(nw.col("price").mean())
            .sort("date")
            .to_native()
        )

    return (monthly_aggregate,)


@app.cell
def _(
    arrow_table,
    duckdb_df,
    monthly_aggregate,
    pandas_df,
    polars_df,
    spark_df,
):
    # pandas
    monthly_aggregate(pandas_df)

    # polars
    monthly_aggregate(polars_df)

    # duckdb
    monthly_aggregate(duckdb_df)

    # pyarrow
    monthly_aggregate(arrow_table)

    # pyspark
    monthly_aggregate(spark_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Bonus - can we generate SQL?

    Narwhals comes with an extra bonus feature: by combining it with [SQLFrame](https://github.com/eakmanrq/sqlframe), we can easily transpiling the Polars API to any major SQL dialect. For example, to translate to the DataBricks SQL dialect, we can do:
    """
    )
    return


@app.cell
def _(monthly_aggregate, pandas_df):
    from sqlframe.duckdb import DuckDBSession

    sqlframe = DuckDBSession()
    sqlframe_df = sqlframe.createDataFrame(pandas_df)
    sqlframe_result = monthly_aggregate(sqlframe_df)
    print(sqlframe_result.sql(dialect="databricks"))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
