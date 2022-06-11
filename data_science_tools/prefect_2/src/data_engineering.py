import json
import pickle

import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from prefect import flow, task
from pydash import py_
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------- #
#                                 Create tasks                                 #
# ---------------------------------------------------------------------------- #


@task
def load_data(data_path: str):
    data_path = abspath(data_path)
    with open(data_path, "r") as file:
        data = json.load(file)
    return data["animals"]


@task
def process_json(animals: list, attributes: list):
    return {
        attribute: [py_.get(animal, attribute) for animal in animals]
        for attribute in attributes
    }


@task
def convert_to_dataframe(data: dict):
    return pd.DataFrame(data)


@task
def save_data(data: pd.DataFrame, save_location: str):
    data.to_csv(abspath(save_location))


@hydra.main(config_path="../config", config_name="process", version_base=None)
@flow
def process_data(config):
    data = load_data(config.data.raw)
    processed = process_json(data, list(config.attributes))
    frame = convert_to_dataframe(processed)
    save_data(frame, config.data.processed)


# ---------------------------------------------------------------------------- #
#                                 Create a flow                                #
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":
    process_data()
