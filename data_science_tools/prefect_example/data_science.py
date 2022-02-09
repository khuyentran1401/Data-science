import numpy as np
import pandas as pd
from prefect import Flow, Parameter, task
from prefect.engine.results import LocalResult


# ---------------------------------------------------------------------------- #
#                                 Create tasks                                 #
# ---------------------------------------------------------------------------- #
@task(log_stdout=True)
def train_model(
    train_x: pd.DataFrame,
    train_y: pd.DataFrame,
    num_train_iter: int,
    learning_rate: float,
) -> np.ndarray:

    num_iter = num_train_iter
    lr = learning_rate
    X = train_x.to_numpy()
    Y = train_y.to_numpy()

    # Add bias to the features
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((bias, X), axis=1)

    weights = []
    # Train one model for each class in Y
    for k in range(Y.shape[1]):
        # Initialise weights
        theta = np.zeros(X.shape[1])
        y = Y[:, k]
        for _ in range(num_iter):
            z = np.dot(X, theta)
            h = _sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            theta -= lr * gradient
        # Save the weights for each model
        weights.append(theta)

    # Print finishing training message
    print("Finish training the model.")

    # Return a joint multi-class model with weights for all classes
    return np.vstack(weights).transpose()


def _sigmoid(z):
    """A helper sigmoid function used by the training and the scoring tasks."""
    return 1 / (1 + np.exp(-z))


@task
def predict(model: np.ndarray, test_x: pd.DataFrame) -> np.ndarray:
    """Task for making predictions given a pre-trained model and a test set."""
    X = test_x.to_numpy()

    # Add bias to the features
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((bias, X), axis=1)

    # Predict "probabilities" for each class
    result = _sigmoid(np.dot(X, model))

    # Return the index of the class with max probability for all samples
    return np.argmax(result, axis=1)


@task(log_stdout=True)
def report_accuracy(predictions: np.ndarray, test_y: pd.DataFrame) -> None:
    """Task for reporting the accuracy of the predictions performed by the
    previous task. Notice that this function has no outputs, except logging.
    """
    # Get true class index
    target = np.argmax(test_y.to_numpy(), axis=1)
    # Calculate accuracy of predictions
    accuracy = np.sum(predictions == target) / target.shape[0]
    # Log the accuracy of the model
    print(f"Model accuracy on test set: {round(accuracy * 100, 2)}")


# ---------------------------------------------------------------------------- #
#                                 Create a flow                                #
# ---------------------------------------------------------------------------- #

with Flow("data-science") as flow:

    train_test_dict = (
        LocalResult(dir="data/processed/Mon_Dec_20_2021_20:55:20")
        .read(location="split_data_output")
        .value
    )

    # Load data
    train_x = train_test_dict["train_x"]
    train_y = train_test_dict["train_y"]
    test_x = train_test_dict["test_x"]
    test_y = train_test_dict["test_y"]

    # Define parameters
    num_train_iter = Parameter("num_train_iter", default=10000)
    learning_rate = Parameter("learning_rate", default=0.01)

    # Define tasks
    model = train_model(train_x, train_y, num_train_iter, learning_rate)
    predictions = predict(model, test_x)
    report_accuracy(predictions, test_y)


flow.run()
