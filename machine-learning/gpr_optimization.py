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
    return (time,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    return C, GaussianProcessRegressor, Matern, WhiteKernel, np, plt


@app.cell
def _(np):
    def black_box_function(x):
        return - (np.sin(3*x) + 0.5 * x)
    return (black_box_function,)


@app.cell
def _(black_box_function, np, plt):
    X = np.linspace(0, 5.5, 1000).reshape(-1, 1)
    y = black_box_function(X)
    plt.plot(X, y)
    plt.title("Black-box function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()
    return X, y


@app.cell
def _(black_box_function, np):
    X_grid = np.linspace(0, 2, 100).reshape(-1, 1)
    y_grid = black_box_function(X_grid)
    x_best = X_grid[np.argmax(y_grid)]
    return


@app.cell
def _(black_box_function, np, time):
    def train(epochs):
        time.sleep(0.1)  # Simulate a slow training step
        return black_box_function(epochs)

    search_space = np.linspace(0, 5, 1000)
    results = []

    start = time.time()
    for x in search_space:
        loss = train(x)
        results.append((x, loss))
    end = time.time()

    print("Best x:", search_space[np.argmin([r[1] for r in results])])
    print("Time taken:", round(end - start, 2), "seconds")
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
def _(X, X_sample, gpr, plt, y, y_sample):
    # Predict across the domain
    mu, std = gpr.predict(X, return_std=True)

    # Plot the result
    plt.figure(figsize=(10, 5))
    plt.plot(X, y, "k--", label="True function")
    plt.plot(X, mu, "b-", label="GPR mean")
    plt.fill_between(X.ravel(), mu - std, mu + std, alpha=0.3, label="Uncertainty")
    plt.scatter(X_sample, y_sample, c="red", label="Samples")
    plt.legend()
    plt.title("Gaussian Process Fit")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()
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
def _(X, X_sample, expected_improvement, gpr, np, plt, y_sample):
    ei = expected_improvement(X, X_sample, y_sample, gpr)

    plt.figure(figsize=(10, 4))
    plt.plot(X, ei, label="Expected Improvement")
    plt.axvline(X[np.argmax(ei)], color="r", linestyle="--", label="Next sample point")
    plt.title("Acquisition Function (Expected Improvement)")
    plt.xlabel("x")
    plt.ylabel("EI(x)")
    plt.legend()
    plt.show()

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
def _(X, X_opt, black_box_function, plt, y_opt):
    # Plot final sampled points
    plt.plot(X, black_box_function(X), "k--", label="True function")
    plt.scatter(X_opt, y_opt, c="red", label="Sampled Points")
    plt.title("Bayesian Optimization with Gaussian Process")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
