import marimo

__generated_with = "0.13.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
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


@app.function
def monthly_aggregate_bad(user_df):
    if hasattr(user_df, "to_pandas"):
        df = user_df.to_pandas()
    elif hasattr(user_df, "toPandas"):
        df = user_df.toPandas()
    elif hasattr(user_df, "_to_pandas"):
        df = user_df._to_pandas()
    return df.resample("MS", on="date")[["price"]].mean()


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Unmaintainable solution: different branches for each library

    This works, but is unfeasibly difficult to test and maintain, especially when also factoring in API changes between different versions of the same library (e.g. pandas `1.*` vs pandas `2.*`).
    """
    )
    return


@app.cell
def _(F):
    import pandas as pd
    import polars as pl
    import duckdb
    import pyspark


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
                user_df.groupBy(F.date_trunc("month", F.col("date")))
                .agg(F.mean("price"))
                .orderBy("date")
            )
        elif isinstance(user_df, duckdb.DuckDBPyRelation):
            result = user_df.aggregate(
                [
                    duckdb.FunctionExpression(
                        "time_bucket",
                        duckdb.ConstantExpression("1 month"),
                        duckdb.FunctionExpression("date"),
                    ).alias("date"),
                    duckdb.FunctionExpression("mean", "price").alias("price"),
                ],
            ).sort("date")
        # TODO: more branches for PyArrow, Dask, etc... :sob:
        return result
    return duckdb, pd, pl


@app.cell
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
def _(mo):
    mo.md(r"""## Demo: let's verify that it works!""")
    return


@app.cell
def _():
    from datetime import datetime

    data = {
        "date": [datetime(2020, 1, 1), datetime(2020, 1, 8), datetime(2020, 2, 3)],
        "price": [1, 4, 3],
    }
    return (data,)


@app.cell
def _(data, monthly_aggregate, pd):
    # pandas
    df_pd = pd.DataFrame(data)
    monthly_aggregate(df_pd)
    return (df_pd,)


@app.cell
def _(data, monthly_aggregate, pl):
    # Polars
    df_pl = pl.DataFrame(data)
    monthly_aggregate(df_pl)
    return


@app.cell
def _(duckdb, monthly_aggregate):
    # DuckDB
    rel = duckdb.sql("""
        from values (timestamp '2020-01-01', 1),
                    (timestamp '2020-01-08', 4),
                    (timestamp '2020-02-03', 3)
                    df(date, price)
        select *
    """)
    monthly_aggregate(rel)
    return


@app.cell
def _(data, monthly_aggregate):
    # PyArrow
    import pyarrow as pa

    tbl = pa.table(data)
    monthly_aggregate(tbl)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Bonus - can we generate SQL?

    Narwhals comes with an extra bonus feature: by combining it with [SQLFrame](https://github.com/eakmanrq/sqlframe), we can easily transpiling the Polars API to any major SQL dialect. For example, to translate to the DataBricks SQL dialect, we can do:
    """
    )
    return


@app.cell
def _(df_pd, monthly_aggregate):
    from sqlframe.duckdb import DuckDBSession

    sqlframe = DuckDBSession()
    sqlframe_df = sqlframe.createDataFrame(df_pd)
    sqlframe_result = monthly_aggregate(sqlframe_df)
    print(sqlframe_result.sql(dialect="databricks"))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
