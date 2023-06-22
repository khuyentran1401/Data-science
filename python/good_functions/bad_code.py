import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import gdown


def get_data(
    url: str,
    zip_path: str,
    raw_train_path: str,
    raw_test_path: str,
    processed_train_path: str,
    processed_test_path: str,
):
    # Download data from Google Drive
    gdown.download(url, zip_path, quiet=False)

    # Unzip data
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

    # Extract texts from files in the train directory
    t_train = []
    for file_path in Path(raw_train_path).glob("*.xml"):
        list_train_doc_1 = [r.text for r in ET.parse(file_path).getroot()[0]]
        train_doc_1 = " ".join(t for t in list_train_doc_1)
        t_train.append(train_doc_1)
    t_train_docs = " ".join(t_train)

    # Extract texts from files in the test directory
    t_test = []
    for file_path in Path(raw_test_path).glob("*.xml"):
        list_test_doc_1 = [r.text for r in ET.parse(file_path).getroot()[0]]
        test_doc_1 = " ".join(t for t in list_test_doc_1)
        t_test.append(test_doc_1)
    t_test_docs = " ".join(t_test)

    # Write processed data to a train file
    with open(processed_train_path, "w") as f:
        f.write(t_train_docs)

    # Write processed data to a test file
    with open(processed_test_path, "w") as f:
        f.write(t_test_docs)


if __name__ == "__main__":
    get_data(
        url="https://drive.google.com/uc?id=1jI1cmxqnwsmC-vbl8dNY6b4aNBtBbKy3",
        zip_path="Twitter.zip",
        raw_train_path="Data/train/en",
        raw_test_path="Data/test/en",
        processed_train_path="Data/train/en.txt",
        processed_test_path="Data/test/en.txt",
    )
