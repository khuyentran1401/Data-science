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
    import marimo as mo
    from marimo import ui

    multiplier = ui.slider(1, 10, 1, label="Multiplier")
    multiplier
    return mo, multiplier


@app.cell
def _(mo, multiplier):
    stars = "⭐" * multiplier.value
    mo.md(stars)
    return


if __name__ == "__main__":
    app.run()
