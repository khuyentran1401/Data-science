import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from marimo import ui

    multiplier = ui.slider(1, 10, 3, label="Multiplier")
    multiplier
    return (multiplier,)


@app.cell
def _(multiplier):
    result = [x * multiplier.value for x in range(5)]
    print(result)
    return


if __name__ == "__main__":
    app.run()
