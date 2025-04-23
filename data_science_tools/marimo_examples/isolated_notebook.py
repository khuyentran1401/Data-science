# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy==2.2.5",
#     "pandas==2.2.3",
# ]
# ///


import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd

    np.random.seed(1)
    df = pd.DataFrame({"value": np.random.randn(5)})
    print(df)
    return


if __name__ == "__main__":
    app.run()
