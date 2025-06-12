# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.13.7"
app = marimo.App(width="medium")


@app.cell
def _():
    threshold = 30
    return (threshold,)


@app.cell
def _(threshold):
    data = [20, 40, 60, 80]
    filtered = [x for x in data if x > threshold]
    print(filtered)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
