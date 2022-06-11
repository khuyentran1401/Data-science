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
def load_data():
    ...


@flow
def process_data():
    ...


# ---------------------------------------------------------------------------- #
#                                 Create a flow                                #
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":
    process_data()
