# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==2.2.5",
#     "pandas==2.2.3",
#     "polars==1.27.1",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.13.0"
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
    df = pd.DataFrame(data)
    df.head(10)
    return df, pd


@app.cell
def _(df):
    df.to_csv("large_file.csv", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Multi-Core Performance""")
    return


@app.cell
def _():
    import time
    from functools import wraps

    def timeit(operation_name):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"{operation_name} took {execution_time:.2f} seconds")
                return result, execution_time

            return wrapper

        return decorator

    return (timeit,)


@app.cell
def create_comparison_plot():
    import matplotlib.pyplot as plt
    import seaborn as sns

    def create_comparison_plot(pandas_time, polars_time, title):
        # Set style for this plot
        sns.set(style="whitegrid")
        plt.style.use("dark_background")
        plt.rcParams.update(
            {
                "axes.facecolor": "#2F2D2E",
                "figure.facecolor": "#2F2D2E",
                "axes.labelcolor": "white",
                "xtick.color": "white",
                "ytick.color": "white",
                "text.color": "white",
            }
        )

        # Create the plot
        sns.barplot(
            hue=["Pandas", "Polars"],
            y=[pandas_time, polars_time],
            palette=["#E583B6", "#72BEFA"],
        )
        plt.title(f"{title} (seconds)")
        plt.ylabel("Time (s)")
        plt.show()

    return (create_comparison_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Reading Data""")
    return


@app.cell
def _(pd, timeit):
    @timeit("Pandas read_csv")
    def read_pandas():
        return pd.read_csv("large_file.csv")

    pandas_df, pandas_read_time = read_pandas()
    return pandas_df, pandas_read_time


@app.cell
def _(timeit):
    import polars as pl

    @timeit("Polars read_csv")
    def read_polars():
        return pl.read_csv("large_file.csv")

    polars_df, polars_read_time = read_polars()
    return pl, polars_df, polars_read_time


@app.cell
def _(create_comparison_plot, pandas_read_time, polars_read_time):
    create_comparison_plot(pandas_read_time, polars_read_time, "CSV Read Time")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Groupby Mean""")
    return


@app.cell
def _(pandas_df, timeit):
    @timeit("Pandas groupby")
    def pandas_groupby(df):
        return df.groupby("category")["value"].mean()

    pandas_result, pandas_groupby_time = pandas_groupby(pandas_df)
    return (pandas_groupby_time,)


@app.cell
def _(pl, polars_df, timeit):
    @timeit("Polars groupby")
    def polars_groupby(df):
        return df.group_by("category").agg(pl.col("value").mean())

    polars_result, polars_groupby_time = polars_groupby(polars_df)
    return (polars_groupby_time,)


@app.cell
def _(create_comparison_plot, pandas_groupby_time, polars_groupby_time):
    create_comparison_plot(
        pandas_groupby_time, polars_groupby_time, "Groupby Mean Time"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Filter Rows""")
    return


@app.cell
def _(pandas_df, timeit):
    @timeit("Pandas filter")
    def pandas_filter(df):
        return df[df["value"] > 500]

    pandas_filtered, pandas_filter_time = pandas_filter(pandas_df)
    return (pandas_filter_time,)


@app.cell
def _(pl, polars_df, timeit):
    @timeit("Polars filter")
    def polars_filter(df):
        return df.filter(pl.col("value") > 500)

    polars_filtered, polars_filter_time = polars_filter(polars_df)
    return (polars_filter_time,)


@app.cell
def _(create_comparison_plot, pandas_filter_time, polars_filter_time):
    create_comparison_plot(pandas_filter_time, polars_filter_time, "Filter Rows Time")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Sort by Column""")
    return


@app.cell
def _(pandas_df, timeit):
    @timeit("Pandas sort")
    def pandas_sort(df):
        return df.sort_values("value")

    pandas_sorted, pandas_sort_time = pandas_sort(pandas_df)
    return (pandas_sort_time,)


@app.cell
def _(polars_df, timeit):
    @timeit("Polars sort")
    def polars_sort(df):
        return df.sort("value")

    polars_sorted, polars_sort_time = polars_sort(polars_df)
    return (polars_sort_time,)


@app.cell
def _(create_comparison_plot, pandas_sort_time, polars_sort_time):
    create_comparison_plot(pandas_sort_time, polars_sort_time, "Sort Time")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Lazy Evaluation (Only in Polars)""")
    return


@app.cell
def _(pl, polars_df):
    query = (
        polars_df.lazy()
        .group_by("category")
        .agg(pl.col("value").mean().alias("avg_value"))
        .filter(pl.col("avg_value") > 100)
    )

    print(query.explain())
    return (query,)


@app.cell
def _(query):
    result = query.collect()
    print(result.head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Syntax Comparison""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Filtering rows""")
    return


@app.cell
def _(pandas_df):
    pandas_filtered_rows_df = pandas_df[pandas_df["value"] > 100]
    return


@app.cell
def _(pl, polars_df):
    polars_filtered_rows_df = polars_df.filter(pl.col("value") > 100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Selecting columns""")
    return


@app.cell
def _(pandas_df):
    pandas_selected_columns_df = pandas_df[["category", "value"]]
    return


@app.cell
def _(polars_df):
    polars_selected_columns_df = polars_df.select(["category", "value"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Chained operations""")
    return


@app.cell
def _(pandas_df):
    pandas_chained_operations_df = pandas_df[pandas_df["value"] > 1000]
    pandas_chained_operations_df = (
        pandas_chained_operations_df.groupby("category")["value"].mean().reset_index()
    )
    return


@app.cell
def _(pl, polars_df):
    polars_chained_operations_df = polars_df.filter(pl.col("value") > 1000)
    polars_chained_operations_df = polars_chained_operations_df.group_by(
        "category"
    ).agg(pl.col("value").mean().alias("avg_value"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Memory Efficiency""")
    return


@app.cell
def _(pandas_df, polars_df):
    pandas_memory = pandas_df.memory_usage(deep=True).sum() / 1000000.0
    polars_memory = polars_df.estimated_size() / 1000000.0

    print(f"Pandas DataFrame memory usage: {pandas_memory:.2f} MB")
    print(f"Polars DataFrame estimated size: {polars_memory:.2f} MB")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
