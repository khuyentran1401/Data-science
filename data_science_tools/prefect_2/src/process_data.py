import pickle

import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from prefect import flow, task
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None
# ---------------------------------------------------------------------------- #
#                                 Create tasks                                 #
# ---------------------------------------------------------------------------- #


@task
def get_data(data_path: str):
    return pd.read_csv(abspath(data_path))


def fill_na_description(data: pd.DataFrame):
    data["Description"] = data["Description"].fillna("")
    return data


def get_desc_length(data: pd.DataFrame):
    data["desc_length"] = data.apply(lambda x: len(x))
    return data


def get_desc_words(data: pd.DataFrame):
    data["desc_words"] = data["Description"].apply(lambda x: len(x.split()))
    return data


def get_average_word_length(data: pd.DataFrame):
    data["average_word_length"] = data["desc_length"] / data["desc_words"]
    return data


@task
def get_description_features(data: pd.DataFrame):
    return (
        data.pipe(fill_na_description)
        .pipe(get_desc_length)
        .pipe(get_desc_words)
        .pipe(get_average_word_length)
    )


@task
def filter_cols(use_cols: list, data: pd.DataFrame):
    return data[use_cols]


@task
def encode_cat_cols(cat_cols: list, data: pd.DataFrame):
    cat_cols = list(cat_cols)
    data[cat_cols] = data[cat_cols].astype(str)
    for col in cat_cols:
        _, indexer = pd.factorize(data[col])
        data[col] = indexer.get_indexer(data[col])
    return data


def split_X_y(data: pd.DataFrame):
    X = data.drop(columns=["AdoptionSpeed"])
    y = data["AdoptionSpeed"]
    return X, y


@task
def split_data(data: pd.DataFrame):
    train = data.dropna(subset=["AdoptionSpeed"])
    test = data[data["AdoptionSpeed"].isna()]
    X_train, y_train = split_X_y(train)
    X_test, _ = split_X_y(test)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train}


@task
def save_data(data: dict, save_dir: str):

    for name, value in data.items():
        save_path = abspath(save_dir + name + ".csv")
        value.to_csv(save_path)


@hydra.main(config_path="../config", config_name="process", version_base=None)
@flow
def process_data(config):
    data = get_data(config.data.raw.path)
    processed = get_description_features(data)
    filtered = filter_cols(config.use_cols, processed)
    encoded = encode_cat_cols(config.cat_cols, filtered)
    split = split_data(encoded)
    save_data(split, config.data.processed)


# ---------------------------------------------------------------------------- #
#                                 Create a flow                                #
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":
    process_data()
