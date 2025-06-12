# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
# ]
# ///
import marimo

__generated_with = "0.13.0"
app = marimo.App()


@app.cell
def _():
    data = [1, 2, 3]
    return (data,)


@app.cell
def _(data):
    summary = sum(data)
    print("Sum:", summary)
    return


@app.cell
def _():
    data_1 = [10, 20, 30]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
