# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "narwhals==1.36.0",
#     "pandas==2.2.3",
#     "pyspark==3.5.5",
# ]
# ///

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Set Up""")
    return


@app.cell
def _():
    from datetime import date

    import pandas as pd
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    return date, pd, spark


@app.cell
def _(date, pd, spark):
    # Create a Spark DataFrame
    item_price_pandas = pd.DataFrame(
        {
            "item_id": [1, 2, 3, 4],
            "price": [4, 2, 5, 1],
            "transaction_date": [
                date(2025, 1, 15),
                date(2025, 2, 1),
                date(2025, 3, 10),
                date(2025, 4, 22),
            ],
        }
    )

    item_price = spark.createDataFrame(item_price_pandas)
    item_price.show()
    return (item_price,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Traditional Query Approach""")
    return


@app.cell
def _(item_price, spark):
    item_price.createOrReplaceTempView("item_price_view")
    transaction_date_str = "2025-02-15"

    query_with_fstring = f"""SELECT *
    FROM item_price_view
    WHERE transaction_date > '{transaction_date_str}'
    """

    spark.sql(query_with_fstring).show()
    return (transaction_date_str,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Parameterized Queries with PySpark Custom String Formatting""")
    return


@app.cell
def _(item_price, spark, transaction_date_str):
    parametrized_query = """SELECT *
    FROM {item_price}
    WHERE transaction_date > {transaction_date}
    """

    spark.sql(
        parametrized_query, item_price=item_price, transaction_date=transaction_date_str
    ).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Parameterized Queries with Parameter Markers""")
    return


@app.cell
def _(date, item_price, spark):
    query_with_markers = """SELECT *
    FROM {item_price}
    WHERE transaction_date > :transaction_date
    """

    transaction_date = date(2025, 2, 15)

    spark.sql(
        query_with_markers,
        item_price=item_price,
        args={"transaction_date": transaction_date},
    ).show()
    return (query_with_markers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Make SQL Easier to Reuse""")
    return


@app.cell
def _(date, item_price, query_with_markers, spark):
    transaction_date_1 = date(2025, 3, 9)

    spark.sql(
        query_with_markers,
        item_price=item_price,
        args={"transaction_date": transaction_date_1},
    ).show()
    return


@app.cell
def _(date, item_price, query_with_markers, spark):
    transaction_date_2 = date(2025, 3, 15)

    spark.sql(
        query_with_markers,
        item_price=item_price,
        args={"transaction_date": transaction_date_2},
    ).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Easier Unit Testing with Parameterized Queries""")
    return


@app.cell
def _(spark):
    def filter_by_price_threshold(df, amount):
        return spark.sql(
            "SELECT * from {df} where price > :amount", df=df, args={"amount": amount}
        )

    return (filter_by_price_threshold,)


@app.cell
def test_query_return_correct_number_of_rows(filter_by_price_threshold, spark):
    # Create test input DataFrame
    df = spark.createDataFrame(
        [
            ("Product 1", 10.0, 5),
            ("Product 2", 15.0, 3),
            ("Product 3", 8.0, 2),
        ],
        ["name", "price", "quantity"],
    )

    # Execute query with parameters
    assert filter_by_price_threshold(df, 10).count() == 1
    assert filter_by_price_threshold(df, 8).count() == 2
    return


@app.cell(hide_code=True)
def imports():
    import pytest

    return


if __name__ == "__main__":
    app.run()
