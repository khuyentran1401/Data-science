import mlfoundry as mlf
import numpy as np
import pandas as pd
import shap
from mlfoundry.mlfoundry_run import MlFoundryRun
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


def train_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    n_neighbors: int,
    mlf_run: MlFoundryRun,
) -> np.ndarray:

    X = train_x.to_numpy()
    Y = train_y.to_numpy()

    # Create a new model instance
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the model
    knn.fit(X, Y)

    # Print finishing training message
    print("Finish training the model.")

    # Log model
    mlf_run.log_model(knn, mlf.ModelFramework.SKLEARN)

    return knn


def predict(model: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    """Make predictions given a pre-trained model and a test set."""
    X = X.to_numpy()

    return {"predictions": model.predict(X)}


def get_shap_values(model, X_train: pd.DataFrame, X_test: pd.DataFrame):
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    return explainer.shap_values(X_test)


def log_data_stats(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_y: pd.Series,
    test_y: pd.Series,
    model,
    mlf_run: MlFoundryRun,
):
    prediction_train = pd.DataFrame(predict(model, train_x))
    prediction_test = pd.DataFrame(predict(model, test_x))

    train_data = pd.concat([train_x, train_y], axis=1).reset_index(drop=True)
    test_data = pd.concat([test_x, test_y], axis=1).reset_index(drop=True)

    # Log data
    mlf_run.log_dataset(train_data, data_slice=mlf.DataSlice.TRAIN)
    mlf_run.log_dataset(test_data, data_slice=mlf.DataSlice.TEST)

    # Concat data and predictions
    train_df = pd.concat(
        [
            train_data,
            prediction_train,
        ],
        axis=1,
    )
    test_df = pd.concat(
        [
            test_data,
            prediction_test,
        ],
        axis=1,
    )

    # Get SHAP values
    shap_values = get_shap_values(model, train_x, test_x)

    # Log dataset stats
    data_schema = mlf.Schema(
        feature_column_names=list(train_df.columns),
        actual_column_name="species",
        prediction_column_name="predictions",
    )

    mlf_run.log_dataset_stats(
        train_df,
        data_slice=mlf.DataSlice.TRAIN,
        data_schema=data_schema,
        model_type=mlf.ModelType.MULTICLASS_CLASSIFICATION,
        shap_values=shap_values,
    )

    mlf_run.log_dataset_stats(
        test_df,
        data_slice=mlf.DataSlice.TEST,
        data_schema=data_schema,
        model_type=mlf.ModelType.MULTICLASS_CLASSIFICATION,
        shap_values=shap_values,
    )

    log_metrics(prediction_test["predictions"], test_y, mlf_run)


def log_metrics(
    predictions: np.ndarray, test_y: pd.DataFrame, mlf_run: MlFoundryRun
) -> None:

    target = test_y.to_numpy()

    # Get metrics
    metrics = {}
    metrics["accuracy"] = accuracy_score(target, predictions)
    metrics["f1_score"] = f1_score(target, predictions, average="weighted")

    # Log metrics
    mlf_run.log_metrics(metrics)


# ---------------------------------------------------------------------------- #
#                                 Create a flow                                #
# ---------------------------------------------------------------------------- #


def data_science_flow(train_test_dict: dict, mlf_run: MlFoundryRun):

    # Load data
    train_x = train_test_dict["train_x"]
    train_y = train_test_dict["train_y"]
    test_x = train_test_dict["test_x"]
    test_y = train_test_dict["test_y"]

    # Define parameters
    params = {"n_neighbors": 12}

    # Log parameters
    mlf_run.log_params(params)

    # Define tasks
    model = train_model(train_x, train_y, params["n_neighbors"], mlf_run)

    log_data_stats(train_x, test_x, train_y, test_y, model, mlf_run)
