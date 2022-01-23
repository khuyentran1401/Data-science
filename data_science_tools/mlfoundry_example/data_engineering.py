from typing import Any, Dict, List

import mlfoundry as mlf
import pandas as pd
from mlfoundry.mlfoundry_run import MlFoundryRun
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def encode_categorical_columns(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Task for encoding the categorical columns in the Iris data set."""

    enc = OrdinalEncoder()
    data[target_col] = enc.fit_transform(data[[target_col]])
    return data


def split_data(
    data: pd.DataFrame, target_col: str, test_data_ratio: float, mlf_run: MlFoundryRun
) -> Dict[str, Any]:
    """Task for splitting the classical Iris data set into training and test
    sets, each split into features and labels.
    """

    print(f"Splitting data into training and test sets with ratio {test_data_ratio}")

    X, y = data.drop(columns=target_col), data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data_ratio)

    # When returning many variables, it is a good practice to give them names:
    return dict(
        train_x=X_train,
        train_y=y_train,
        test_x=X_test,
        test_y=y_test,
    )


def data_engineer_flow(mlf_run: MlFoundryRun):

    # Define parameters
    params = {"target_col": "species", "test_data_ratio": 0.2}

    # Log parameters
    mlf_run.log_params(params)

    # Define tasks
    data = load_data(path="data/raw/iris.csv")
    categorical_columns = encode_categorical_columns(
        data=data, target_col=params["target_col"]
    )
    train_test_dict = split_data(
        data=categorical_columns,
        target_col=params["target_col"],
        test_data_ratio=params["test_data_ratio"],
        mlf_run=mlf_run,
    )

    return train_test_dict
