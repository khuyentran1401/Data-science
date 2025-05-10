import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("# Write a markdown header")
    return


@app.cell
def _():
    # Cell 1: Define a variable
    message = "Hi"
    print(message)
    return (message,)


@app.cell
def _(message):
    # Cell 2: Reference the variable from Cell 1
    print(f"I'm using the message: {message}")
    return


@app.cell
def _(mo):
    dropdown = mo.ui.dropdown(["1", "2", "3"], value="1")
    dropdown
    return (dropdown,)


@app.cell
def _(mo):
    slider = mo.ui.slider(1, 16, value=8, label="Number of numbers")
    slider
    return (slider,)


@app.cell
def _(mo):
    icon = mo.ui.dropdown(["🍃", "🌊", "✨"], value="🍃")
    return (icon,)


@app.cell
def _(icon, mo):
    repetitions = mo.ui.slider(1, 16, label=f"number of {icon.value}: ")
    return (repetitions,)


@app.cell
def _(icon, repetitions):
    icon, repetitions
    return


@app.cell
def _(icon, mo, repetitions):
    mo.md("# " + icon.value * repetitions.value)
    return


@app.cell
def _(np):
    class DataProcessor:
        def __init__(self, data):
            self._data = data
    
        def transform(self):
            return self._data * 2
    
        def summarize(self):
            return {"mean": self._data.mean(), "std": self._data.std()}

    # Create an instance
    processor = DataProcessor(np.random.randn(100))
    return DataProcessor, processor


@app.cell
def _(mo, processor):
    results = processor.summarize()
    mo.md(f"""
    ## Data Summary
    - Mean: {results['mean']:.4f}
    - Standard Deviation: {results['std']:.4f}
    """)
    return (results,)


@app.cell
def _(np):
    _temp_data = np.random.randn(50)
    result_1 = _temp_data.mean()
    print(f"Result 1: {result_1:.4f}")
    return (result_1,)


@app.cell
def _(np):
    # Cell 2 (can use _temp_data again)
    _temp_data = np.random.randn(100)  # Different from Cell 1's _temp_data
    result_2 = _temp_data.mean()
    print(f"Result 2: {result_2:.4f}")
    return (result_2,)


@app.cell
def _():
    import numpy as np
    return (np,)


if __name__ == "__main__":
    app.run()
