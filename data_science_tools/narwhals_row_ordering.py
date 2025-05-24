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
def _():
    from datetime import datetime

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
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Eager-only solution""")
    return


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
def _(agnostic_ffill_by_store, data):
    import pandas as pd
    import polars as pl

    # pandas.DataFrame
    df_pandas = pd.DataFrame(data)
    agnostic_ffill_by_store(df_pandas)

    # polars.DataFrame
    df_polars = pl.DataFrame(data)
    agnostic_ffill_by_store(df_polars)
    return (df_pandas,)


@app.cell
def _():
    import duckdb

    duckdb_rel = duckdb.table("df_polars")
    duckdb_rel
    return (duckdb_rel,)


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
def _(agnostic_ffill_by_store_improved, df_pandas):
    # Note that it still supports pandas
    agnostic_ffill_by_store_improved(df_pandas)
    return


if __name__ == "__main__":
    app.run()
