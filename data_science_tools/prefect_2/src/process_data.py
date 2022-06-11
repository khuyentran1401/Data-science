import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from nltk.tokenize import TweetTokenizer
from prefect import flow, task
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------- #
#                                 Create tasks                                 #
# ---------------------------------------------------------------------------- #


@task
def get_data(data_path: str):
    train = pd.read_csv(abspath(data_path.train))
    test = pd.read_csv(abspath(data_path.test))
    return {"train": train, "test": test}


@task
def get_all_data(data: dict):
    return pd.concat([data["train"], data["test"]])


@task
def get_vectorizer(data: pd.DataFrame):
    tokenizer = TweetTokenizer()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)
    vectorizer.fit(data["Description"].fillna("").values)
    return vectorizer


@task
def encode_description(vectorizer: TfidfVectorizer, data: pd.DataFrame):
    X_train = vectorizer.transform(data["Description"].fillna(""))
    print(X_train)
    print(type(X_train))
    return X_train


@task
def get_adoption_speed(data: pd.DataFrame):
    return data["AdoptionSpeed"]


@task
def get_classifier(data: pd.DataFrame, adoption_speed: pd.Series, n_estimators: int):
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(data, adoption_speed)


@flow
def get_description_features(config, all_data, data: dict):
    vectorizer = get_vectorizer(all_data)
    X_train = encode_description(vectorizer, data["train"])
    y_train = get_adoption_speed


@hydra.main(config_path="../config", config_name="process", version_base=None)
@flow
def process_data(config):
    data = get_data(config.data.raw)
    all_data = get_all_data(data)
    get_description_features(config, all_data, data)


# ---------------------------------------------------------------------------- #
#                                 Create a flow                                #
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":
    process_data()
