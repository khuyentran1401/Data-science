# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.2.6",
#     "scikit-learn==1.6.1",
#     "scipy==1.15.3",
# ]
# ///

import marimo

__generated_with = "0.13.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt

    def apply_codecut_style(ax=None):
        """
        Apply CodeCut plot styling to a given Matplotlib Axes.
        If no Axes is provided, use the current active Axes.
        """
        if ax is None:
            ax = plt.gca()

        # Set global figure facecolor
        plt.figure(facecolor="#2F2D2E")

        # Background colors
        fig = ax.figure
        fig.patch.set_facecolor("#2F2D2E")
        ax.set_facecolor("#2F2D2E")

        # Line and text colors
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")

        # Spine colors
        for spine in ax.spines.values():
            spine.set_color("white")

        # Optional: turn off grid
        ax.grid(False)

        return ax
    return apply_codecut_style, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Motivation""")
    return


@app.cell
def _():
    import random
    import time

    def train_model(epochs, batch_size):
        # Simulate training by producing a score based on epochs and batch size
        time.sleep(0.5)  # 0.5 second delay to mimic compute time
        random.seed(epochs + batch_size)
        return {"score": random.uniform(0.7, 0.95)}

    def evaluate_model(model):
        return model["score"]

    best_score = float("-inf")
    best_params = None

    for epochs in [10, 50, 100]:
        for batch_size in [16, 32, 64]:
            print(f"Training model with epochs={epochs}, batch_size={batch_size}...")
            model = train_model(epochs=epochs, batch_size=batch_size)
            score = evaluate_model(model)
            print(f"--> Score: {score:.4f}")
            if score > best_score:
                best_score = score
                best_params = {"epochs": epochs, "batch_size": batch_size}
                print(f"--> New best score! Updated best_params: {best_params}")

    print("Best score:", best_score)
    print("Best params:", best_params)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Setup""")
    return


@app.cell
def _():
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    return C, GaussianProcessRegressor, Matern, WhiteKernel, np


@app.cell
def _(np):
    def black_box_function(x):
        return - (np.sin(3*x) + 0.5 * x)
    return (black_box_function,)


@app.cell
def _(black_box_function, np):
    X = np.linspace(0, 5.5, 1000).reshape(-1, 1)
    y = black_box_function(X)
    return X, y


@app.cell
def _(X, apply_codecut_style, plt, y):
    plt.plot(X, y, "--", color="white")
    plt.title("Black-box function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    apply_codecut_style()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Brute force hyperparameter search""")
    return


@app.cell
def _(black_box_function, np):
    X_grid = np.linspace(0, 2, 100).reshape(-1, 1)
    y_grid = black_box_function(X_grid)
    x_best = X_grid[np.argmax(y_grid)]
    return X_grid, x_best, y_grid


@app.cell
def _(X_grid, apply_codecut_style, black_box_function, plt, x_best, y_grid):
    plt.plot(X_grid, y_grid, "--", color="white", label="True function")
    plt.scatter(X_grid, y_grid, c="#E583B6", label="Evaluated Points")
    plt.scatter(x_best, black_box_function(x_best), c="#72BEFA", s=80, edgecolors="black", label="Best Point")

    plt.title("Brute Force Search Over Full Range")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    apply_codecut_style()
    return


@app.cell
def _():
    # def train(epochs):
    #     time.sleep(0.1)  # Simulate a slow training step
    #     return black_box_function(epochs)

    # search_space = np.linspace(0, 5, 1000)
    # results = []

    # start = time.time()
    # for x in search_space:
    #     loss = train(x)
    #     results.append((x, loss))
    # end = time.time()

    # best_x = search_space[np.argmin([r[1] for r in results])]
    # print(f"Best x: {best_x}")
    # print("Time taken:", round(end - start, 2), "seconds")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Introducing Gaussian Process Regression""")
    return


@app.cell
def _(black_box_function, np):
    # Initial sample points (simulate prior evaluations)
    X_sample = np.array([[1.0], [3.0], [5.5]])
    y_sample = black_box_function(X_sample)
    return X_sample, y_sample


@app.cell
def _(C, GaussianProcessRegressor, Matern, WhiteKernel, X_sample, y_sample):
    # Define the kernel
    kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))

    # Create and fit the Gaussian Process model
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    gpr.fit(X_sample, y_sample)
    return (gpr,)


@app.cell
def _(X, X_sample, apply_codecut_style, gpr, plt, y, y_sample):
    # Predict across the domain
    mu, std = gpr.predict(X, return_std=True)

    # Plot the result
    plt.figure(figsize=(10, 5))
    plt.plot(X, y, "--", label="True function", color="white")
    plt.plot(X, mu, "-", label="GPR mean", color="#72BEFA")
    plt.fill_between(X.ravel(), mu - std, mu + std, alpha=0.3, label="Uncertainty")
    plt.scatter(X_sample, y_sample, c="#E583B6", label="Samples")
    plt.legend()
    plt.title("Gaussian Process Fit")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    apply_codecut_style()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Bayesian Optimization Step""")
    return


@app.cell
def _(np):
    from scipy.stats import norm

    def expected_improvement(X, X_sample, y_sample, model, xi=0.01):
        mu, std = model.predict(X, return_std=True)
        mu_sample_opt = np.min(y_sample)

        with np.errstate(divide="warn"):
            imp = mu_sample_opt - mu - xi  # because we are minimizing
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0

        return ei

    return (expected_improvement,)


@app.cell
def _(
    X,
    X_sample,
    apply_codecut_style,
    expected_improvement,
    gpr,
    np,
    plt,
    y_sample,
):
    ei = expected_improvement(X, X_sample, y_sample, gpr)

    plt.figure(figsize=(10, 4))
    plt.plot(X, ei, label="Expected Improvement", color="#72BEFA")
    plt.axvline(X[np.argmax(ei)], color="#E583B6", linestyle="--", label="Next sample point")
    plt.title("Acquisition Function (Expected Improvement)")
    plt.xlabel("x")
    plt.ylabel("EI(x)")
    plt.legend()
    apply_codecut_style()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Hypeparameter Search Loop""")
    return


@app.cell
def _(X, black_box_function, expected_improvement, gpr, np):
    def bayesian_optimization(n_iter=10):
        # Initial data
        X_sample = np.array([[1.0], [2.5], [4.0]])
        y_sample = black_box_function(X_sample)

        for _ in range(n_iter):
            gpr.fit(X_sample, y_sample)
            ei = expected_improvement(X, X_sample, y_sample, gpr)
            x_next = X[np.argmax(ei)].reshape(-1, 1)

            # Evaluate the function at the new point
            y_next = black_box_function(x_next)

            # Add the new sample to our dataset
            X_sample = np.vstack((X_sample, x_next))
            y_sample = np.append(y_sample, y_next)
        return X_sample, y_sample

    return (bayesian_optimization,)


@app.cell
def _(bayesian_optimization):
    X_opt, y_opt = bayesian_optimization(n_iter=10)

    return X_opt, y_opt


@app.cell
def _(X, X_opt, apply_codecut_style, black_box_function, plt, y_opt):
    # Plot final sampled points
    plt.plot(X, black_box_function(X), "--", label="True function", color="white")
    plt.scatter(X_opt, y_opt, c="#E583B6", label="Sampled Points")
    plt.title("Bayesian Optimization with Gaussian Process")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    apply_codecut_style()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
