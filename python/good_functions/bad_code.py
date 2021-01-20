import xml.etree.ElementTree as ET
import zipfile
from os import listdir
from os.path import isfile, join

import gdown


def main():

    load_data(
        url="https://drive.google.com/uc?id=1jI1cmxqnwsmC-vbl8dNY6b4aNBtBbKy3",
        output="Twitter.zip",
        path_train="Data/train/en",
        path_test="Data/test/en",
    )


def load_data(url: str, output: str, path_train: str, path_test: str):

    # Download data from Google Drive
    output = "Twitter.zip"
    gdown.download(url, output, quiet=False)

    # Unzip data
    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall(".")

    # Get train, test data files
    tweets_train_files = [
        file
        for file in listdir(path_train)
        if isfile(join(path_train, file)) and file != "truth.txt"
    ]
    tweets_test_files = [
        file
        for file in listdir(path_test)
        if isfile(join(path_test, file)) and file != "truth.txt"
    ]

    # Extract texts from each file
    t_train = []
    for file in tweets_train_files:
        train_doc_1 = [r.text for r in ET.parse(join(path_train, file)).getroot()[0]]
        t_train.append(" ".join(t for t in train_doc_1))

    t_test = []
    for file in tweets_test_files:
        test_doc_1 = [r.text for r in ET.parse(join(path_test, file)).getroot()[0]]
        t_test.append(" ".join(t for t in test_doc_1))

    return t_train, t_test


if __name__ == "__main__":
    main()
