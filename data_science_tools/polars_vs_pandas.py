import marimo

__generated_with = "0.12.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Polars vs. Pandas: A Fast, Multi-Core Alternative for DataFrames""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Setup""")
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    # Create a large dataset
    n_rows = 10_000_000
    data = {
        "category": np.random.choice(["A", "B", "C", "D"], size=n_rows),
        "value": np.random.rand(n_rows) * 1000,
    }
    pandas_df = pd.DataFrame(data)
    pandas_df.head(10)
    return data, n_rows, np, pandas_df, pd


@app.cell
def _(pandas_df):
    pandas_df.to_csv("large_file.csv", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Reading Data Faster""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Pandas""")
    return


@app.cell
def _(pd):
    import time

    start_read_pd = time.time()
    df_pd = pd.read_csv("large_file.csv")
    end_read_pd = time.time()
    print(f"Pandas read_csv took {end_read_pd - start_read_pd:.2f} seconds")
    return df_pd, end_read_pd, start_read_pd, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Polars""")
    return


@app.cell
def _(time):
    import polars as pl

    start_read_pl = time.time()
    polars_df = pl.read_csv("large_file.csv")
    end_read_pl = time.time()
    print(f"Polars read_csv took {end_read_pl - start_read_pl:.2f} seconds")
    return end_read_pl, pl, polars_df, start_read_pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. Lazy Evaluation (Only in Polars)""")
    return


@app.cell
def _(pl, polars_df):
    lazy_polars_df = polars_df.lazy()
    result = (
        lazy_polars_df.filter(pl.col("value") > 100)
        .group_by("category")
        .agg(pl.col("value").mean().alias("avg_value"))
        .collect()
    )
    result.head(10)
    return lazy_polars_df, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. Multi-Core Performance""")
    return


@app.cell
def _(data, pd, pl):
    pandas_groupby_df = pd.DataFrame(data)
    polars_groupby_df = pl.DataFrame(data)
    return pandas_groupby_df, polars_groupby_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Groupby Mean""")
    return


@app.cell
def _(pandas_groupby_df, time):
    start_groupby_pd = time.time()
    pandas_groupby_df.groupby("category")["value"].mean()
    end_groupby_pd = time.time()
    print(f"Pandas groupby took {end_groupby_pd - start_groupby_pd:.2f} seconds")
    return end_groupby_pd, start_groupby_pd


@app.cell
def _(pl, polars_groupby_df, time):
    start_groupby_pl = time.time()
    polars_groupby_df.group_by("category").agg(pl.col("value").mean())
    end_groupby_pl = time.time()
    print(f"Polars groupby took {end_groupby_pl - start_groupby_pl:.2f} seconds")
    return end_groupby_pl, start_groupby_pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Filter Rows""")
    return


@app.cell
def _(pandas_groupby_df, time):
    start_filter_pd = time.time()
    pandas_filtered_df = pandas_groupby_df[pandas_groupby_df["value"] > 500]
    end_filter_pd = time.time()
    print(f"Pandas filter took {end_filter_pd - start_filter_pd:.2f} seconds")
    return end_filter_pd, pandas_filtered_df, start_filter_pd


@app.cell
def _(pl, polars_groupby_df, time):
    start_filter_pl = time.time()
    polars_filtered_df = polars_groupby_df.filter(pl.col("value") > 500)
    end_filter_pl = time.time()
    print(f"Polars filter took {end_filter_pl - start_filter_pl:.2f} seconds")
    return end_filter_pl, polars_filtered_df, start_filter_pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Sort by Column""")
    return


@app.cell
def _(pandas_groupby_df, time):
    start_sort_pd = time.time()
    pandas_sorted_df = pandas_groupby_df.sort_values("value")
    end_sort_pd = time.time()
    print(f"Pandas sort took {end_sort_pd - start_sort_pd:.2f} seconds")
    return end_sort_pd, pandas_sorted_df, start_sort_pd


@app.cell
def _(polars_groupby_df, time):
    start_sort_pl = time.time()
    polars_sorted_df = polars_groupby_df.sort("value")
    end_sort_pl = time.time()
    print(f"Polars sort took {end_sort_pl - start_sort_pl:.2f} seconds")
    return end_sort_pl, polars_sorted_df, start_sort_pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Join on Key""")
    return


@app.cell
def _(pd, time):
    pandas_df1 = pd.DataFrame({"key": range(5_000_000), "val1": range(5_000_000)})
    pandas_df2 = pd.DataFrame({"key": range(5_000_000), "val2": range(5_000_000)})
    start_join_pd = time.time()
    pandas_joined_df = pd.merge(pandas_df1, pandas_df2, on="key")
    end_join_pd = time.time()
    print(f"Pandas join took {end_join_pd - start_join_pd:.2f} seconds")
    return end_join_pd, pandas_df1, pandas_df2, pandas_joined_df, start_join_pd


@app.cell
def _(pl, time):
    polars_df1 = pl.DataFrame({"key": range(5_000_000), "val1": range(5_000_000)})
    polars_df2 = pl.DataFrame({"key": range(5_000_000), "val2": range(5_000_000)})
    start_join_pl = time.time()
    polars_joined_df = polars_df1.join(polars_df2, on="key", how="inner")
    end_join_pl = time.time()
    print(f"Polars join took {end_join_pl - start_join_pl:.2f} seconds")
    return end_join_pl, polars_df1, polars_df2, polars_joined_df, start_join_pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. Syntax Comparison""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Filtering rows""")
    return


@app.cell
def _(pandas_groupby_df):
    pandas_filtered_rows_df = pandas_groupby_df[pandas_groupby_df["value"] > 100]
    return (pandas_filtered_rows_df,)


@app.cell
def _(pl, polars_groupby_df):
    polars_filtered_rows_df = polars_groupby_df.filter(pl.col("value") > 100)
    return (polars_filtered_rows_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Selecting columns""")
    return


@app.cell
def _(pandas_groupby_df):
    pandas_selected_columns_df = pandas_groupby_df[["category", "value"]]
    return (pandas_selected_columns_df,)


@app.cell
def _(polars_groupby_df):
    polars_selected_columns_df = polars_groupby_df.select(["category", "value"])
    return (polars_selected_columns_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Chained operations""")
    return


@app.cell
def _(pandas_groupby_df):
    pandas_chained_operations_df = pandas_groupby_df[pandas_groupby_df["value"] > 1000]
    pandas_chained_operations_df = (
        pandas_chained_operations_df.groupby("category")["value"].mean().reset_index()
    )
    return (pandas_chained_operations_df,)


@app.cell
def _(pl, polars_groupby_df):
    polars_chained_operations_df = polars_groupby_df.filter(pl.col("value") > 1000)
    polars_chained_operations_df = polars_chained_operations_df.group_by(
        "category"
    ).agg(pl.col("value").mean().alias("avg_value"))
    return (polars_chained_operations_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. Memory Efficiency""")
    return


@app.cell
def _(pandas_groupby_df, polars_groupby_df):
    print(
        f"Pandas DataFrame memory usage: {pandas_groupby_df.memory_usage(deep=True).sum() / 1000000.0:2f} MB"
    )
    print(
        f"Polars DataFrame estimated size: {polars_groupby_df.estimated_size() / 1000000.0} MB"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
