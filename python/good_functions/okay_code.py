import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Tuple

import gdown


def main(
    url: str,
    zip_path: str,
    raw_train_path: str,
    raw_test_path: str,
    processed_train_path: str,
    processed_test_path: str,
) -> None:
    get_raw_data(url, zip_path)
    t_train, t_test = get_train_test_docs(raw_train_path, raw_test_path)
    save_train_test_docs(processed_train_path, processed_test_path, t_train, t_test)


def get_raw_data(url: str, zip_path: str) -> None:
    gdown.download(url, zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")


def get_train_test_docs(raw_train_path: str, raw_test_path: str) -> Tuple[str, str]:
    t_train = extract_texts_from_multiple_files(raw_train_path)
    t_test = extract_texts_from_multiple_files(raw_test_path)
    return t_train, t_test


def extract_texts_from_multiple_files(folder_path: str) -> str:
    all_docs = []
    for file_path in Path(folder_path).glob("*.xml"):
        text_in_one_file = extract_texts_from_each_file(file_path)
        all_docs.append(text_in_one_file)

    return " ".join(all_docs)


def extract_texts_from_each_file(file_path: str) -> str:
    list_of_text_in_one_file = [r.text for r in ET.parse(file_path).getroot()[0]]
    return " ".join(list_of_text_in_one_file)


def save_train_test_docs(
    processed_train_path, processed_test_path, t_train: str, t_test: str
) -> None:
    save_data(processed_train_path, t_train)
    save_data(processed_test_path, t_test)


def save_data(processed_path: str, processed_data: str) -> None:
    with open(processed_path, "w") as f:
        f.write(processed_data)


if __name__ == "__main__":
    main(
        url="https://drive.google.com/uc?id=1jI1cmxqnwsmC-vbl8dNY6b4aNBtBbKy3",
        zip_path="Twitter.zip",
        raw_train_path="Data/train/en",
        raw_test_path="Data/test/en",
        processed_train_path="Data/train/en.txt",
        processed_test_path="Data/test/en.txt",
    )
