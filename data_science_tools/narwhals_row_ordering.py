# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "duckdb==1.3.0",
#     "marimo",
#     "narwhals==1.40.0",
#     "pandas==2.2.3",
#     "polars==1.30.0",
#     "pyarrow==20.0.0",
# ]
# ///

import marimo

__generated_with = "0.13.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Eager vs Lazy DataFrames: One Fix to Make Your Code Work Anywhere""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Motivation""")
    return


@app.cell
def _():
    from datetime import datetime

    import pandas as pd
    import polars as pl

    data1 = {"store": [1, 1, 2], "date_id": [4, 5, 6]}
    data2 = {"store": [1, 2], "sales": [7, 8]}

    pandas_df1 = pd.DataFrame(data1)
    pandas_df2 = pd.DataFrame(data2)

    # The outputs are  the same
    for _ in range(5):
        # Left join
        pandas_df = pd.merge(pandas_df1, pandas_df2, on="store", how="left")

        # Cumulative sum of sales within each store
        pandas_df["cumulative_sales"] = pandas_df.groupby("store")["sales"].cumsum()

        print(pandas_df)
    return data1, data2, datetime, pd, pl


@app.cell
def _(data1, data2, pl):
    polars_df1 = pl.DataFrame(data1).lazy()
    polars_df2 = pl.DataFrame(data2).lazy()

    # The outputs are not the same
    for _ in range(5):
        print(
            polars_df1.join(polars_df2, on="store", how="left")
            .with_columns(cumulative_sales=pl.col("sales").cum_sum().over("store"))
            .collect(engine="streaming")
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Eager-only solution""")
    return


@app.cell
def _(datetime, pd):
    data = {
    	"sale_date": [
    		datetime(2025, 5, 22),
    		datetime(2025, 5, 23),
    		datetime(2025, 5, 24),
    		datetime(2025, 5, 22),
    		datetime(2025, 5, 23),
    		datetime(2025, 5, 24),
    	],
    	"store": [
    		"Thimphu",
    		"Thimphu",
    		"Thimphu",
    		"Paro",
    		"Paro",
    		"Paro",
    	],
    	"sales": [1100, None, 1450, 501, 500, None],
    }

    pdf = pd.DataFrame(data)
    print(pdf)
    return (data,)


@app.cell
def _():
    import narwhals as nw
    from narwhals.typing import IntoFrameT


    def agnostic_ffill_by_store(df_native: IntoFrameT) -> IntoFrameT:
    	# Supports pandas and Polars.DataFrame, but not lazy ones.
    	return (
    		nw.from_native(df_native)
    		.with_columns(
    			nw.col("sales").fill_null(strategy="forward").over("store")
    		)
    		.to_native()
    	)
    return IntoFrameT, agnostic_ffill_by_store, nw


@app.cell
def _(agnostic_ffill_by_store, data, pd):
    # pandas.DataFrame
    df_pandas = pd.DataFrame(data)
    agnostic_ffill_by_store(df_pandas)
    return (df_pandas,)


@app.cell
def _(agnostic_ffill_by_store, data, pl):
    # polars.DataFrame
    df_polars = pl.DataFrame(data)
    agnostic_ffill_by_store(df_polars)
    return (df_polars,)


@app.cell
def _():
    import duckdb

    duckdb_rel = duckdb.table("df_polars")
    duckdb_rel
    return (duckdb_rel,)


@app.cell
def _():
    # agnostic_ffill_by_store(duckdb_rel)
    # Error: narwhals.exceptions.OrderDependentExprError: Order-dependent expressions are not supported for use in LazyFrame.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Eager and lazy solution""")
    return


@app.cell
def _(IntoFrameT, nw):
    def agnostic_ffill_by_store_improved(df_native: IntoFrameT) -> IntoFrameT:
    	return (
    		nw.from_native(df_native)
    		.with_columns(
    			nw.col("sales")
    			.fill_null(strategy="forward")
    			# Note the `order_by` statement
    			.over("store", order_by="sale_date")
    		)
    		.to_native()
    	)
    return (agnostic_ffill_by_store_improved,)


@app.cell
def _(agnostic_ffill_by_store_improved, duckdb_rel):
    agnostic_ffill_by_store_improved(duckdb_rel)
    return


@app.cell
def _(agnostic_ffill_by_store_improved, df_polars):
    agnostic_ffill_by_store_improved(df_polars.lazy()).collect()
    return


@app.cell
def _(agnostic_ffill_by_store_improved, df_pandas):
    # Note that it still supports pandas
    print(agnostic_ffill_by_store_improved(df_pandas))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
