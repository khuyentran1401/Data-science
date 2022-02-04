import numpy as np
import pandas as pd
from prefect import Flow, Parameter, task
from prefect.engine.results import LocalResult
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


# ---------------------------------------------------------------------------- #
#                                 Create tasks                                 #
# ---------------------------------------------------------------------------- #
@task(log_stdout=True)
def train_model(
    train_x: pd.DataFrame,
    train_y: pd.DataFrame,
    n_neighbors: int,
) -> np.ndarray:

    X = train_x.to_numpy()
    Y = train_y.to_numpy()

    # Create a new model instance
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the model
    knn.fit(X, Y)

    # Print finishing training message
    print("Finish training the model.")

    return knn


@task
def predict(model: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    """Task for making predictions given a pre-trained model and a test set."""
    X = X.to_numpy()

    return model.predict(X)


@task(log_stdout=True)
def report_accuracy(predictions: np.ndarray, test_y: pd.DataFrame) -> None:
    """Task for reporting the accuracy of the predictions performed by the
    previous task. Notice that this function has no outputs, except logging.
    """
    target = test_y.to_numpy()

    accuracy = accuracy_score(target, predictions)
    f1 = f1_score(target, predictions, average="weighted")

    # Log the metrics of the model
    print(f"The model accuracy on test set: {round(accuracy * 100, 2)}")
    print(f"The F1-score on test set: {round(f1 * 100, 2)}")


# ---------------------------------------------------------------------------- #
#                                 Create a flow                                #
# ---------------------------------------------------------------------------- #

with Flow("data-science") as flow:

    train_test_dict = (
        LocalResult(dir="data/processed/Tue_Dec_21_2021_16-26-44/")
        .read(location="split_data_output")
        .value
    )

    # Load data
    train_x = train_test_dict["train_x"]
    train_y = train_test_dict["train_y"]
    test_x = train_test_dict["test_x"]
    test_y = train_test_dict["test_y"]

    # Define parameters
    n_neighbors = Parameter("n_neighbors", default=12)

    # Define tasks
    model = train_model(train_x, train_y, n_neighbors=n_neighbors)
    predictions = predict(model, test_x)
    report_accuracy(predictions, test_y)


flow.run()
