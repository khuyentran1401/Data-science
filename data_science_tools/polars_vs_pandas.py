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
    df = pd.DataFrame(data)
    df.head(10)
    return data, df, n_rows, np, pd


@app.cell
def _(df):
    df.to_csv("large_file.csv", index=False)
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

    start = time.time()
    df_pd = pd.read_csv("large_file.csv")
    end = time.time()
    print(f"Pandas read_csv took {end - start:.2f} seconds")
    return df_pd, end, start, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Polars""")
    return


@app.cell
def _(time):
    import polars as pl

    start_1 = time.time()
    df_pl = pl.read_csv("large_file.csv")
    end_1 = time.time()
    print(f"Polars read_csv took {end_1 - start_1:.2f} seconds")
    return df_pl, end_1, pl, start_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. Lazy Evaluation (Only in Polars)""")
    return


@app.cell
def _(df_pl, pl):
    lazy_df = df_pl.lazy()
    result = (
        lazy_df.filter(pl.col("value") > 100)
        .group_by("category")
        .agg(pl.col("value").mean().alias("avg_value"))
        .collect()
    )
    result.head(10)
    return lazy_df, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. Multi-Core Performance""")
    return


@app.cell
def _(data, pd, pl):
    df_pd_1 = pd.DataFrame(data)
    df_pl_1 = pl.DataFrame(data)
    return df_pd_1, df_pl_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Pandas""")
    return


@app.cell
def _(df_pd_1, time):
    start_2 = time.time()
    df_pd_1.groupby("category")["value"].mean()
    end_2 = time.time()
    print(f"Pandas groupby took {end_2 - start_2:.2f} seconds")
    return end_2, start_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Polars""")
    return


@app.cell
def _(df_pl_1, pl, time):
    start_3 = time.time()
    df_pl_1.group_by("category").agg(pl.col("value").mean())
    end_3 = time.time()
    print(f"Polars groupby took {end_3 - start_3:.2f} seconds")
    return end_3, start_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. Syntax Comparison""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Filtering rows""")
    return


@app.cell
def _(df_pd_1):
    df_pd_1[df_pd_1["value"] > 100]
    return


@app.cell
def _(df_pl_1, pl):
    df_pl_1.filter(pl.col("value") > 100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Selecting columns""")
    return


@app.cell
def _(df_pd_1):
    df_pd_1[["category", "value"]]
    return


@app.cell
def _(df_pl_1):
    df_pl_1.select(["category", "value"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Chained operations""")
    return


@app.cell
def _(df_pd_1):
    df_result = df_pd_1[df_pd_1["value"] > 1000]
    df_result = df_result.groupby("category")["value"].mean().reset_index()
    return (df_result,)


@app.cell
def _(df_pl_1, pl):
    df_result_1 = (
        df_pl_1.filter(pl.col("value") > 1000)
        .group_by("category")
        .agg(pl.col("value").mean().alias("avg_value"))
    )
    return (df_result_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. Memory Efficiency""")
    return


@app.cell
def _(df_pd_1, df_pl_1):
    print(df_pd_1.memory_usage(deep=True).sum() / 1000000.0, "MB")
    print(df_pl_1.estimated_size() / 1000000.0, "MB")
    return


@app.cell
def _(mo):
    mo.md(r"""## Final Thoughts""")
    return


@app.cell
def _(df_pl_1):
    df_pd_2 = df_pl_1.to_pandas()
    return (df_pd_2,)


if __name__ == "__main__":
    app.run()
