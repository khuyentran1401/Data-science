import pickle

import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from prefect import flow, task
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


@task
def load_data(save_path: str):
    return pickle.load(open(abspath(save_path), "rb"))


@task
def train_model(params: DictConfig, X_train: pd.DataFrame, y_train: pd.DataFrame):
    params = dict(params)
    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)
    return clf


@hydra.main(config_path="../config", config_name="train_model", version_base=None)
@flow
def train(config):
    data = load_data(config.data.processed).result()
    clf = train_model(config.params, data["X_train"], data["y_train"])


if __name__ == "__main__":
    train()
