import pandas as pd
from sklearn import datasets


from evidently.model_profile import Profile
from evidently.profile_sections import DataDriftProfileSection
from evidently.pipeline.column_mapping import ColumnMapping

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from prefect import Flow, task
from typing import Tuple
import json 

@task 
def load_data() -> pd.DataFrame:
    california = datasets.fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df["target"] = california.target
    return df 

@task 
def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    return train, test

@task 
def create_data_report(train: pd.DataFrame, test: pd.DataFrame)):
    profiler = Profile(sections=[DataDriftProfileSection])
    profiler.calculate(
        train, test, column_mapping=None
    )
    report = profiler.json()
    json_report = json.loads(report)
    
if __name__ == '__main__':

    with Flow("predict-house-price") as flow:
        df = load_data()
        train, test = split_train_test(df)
    
    flow.run()