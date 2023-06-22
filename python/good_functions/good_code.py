import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Tuple

import gdown
from pydantic import BaseModel


class RawLocation(BaseModel):
    url: str
    zip_path: str
    path_train: str
    path_test: str


class ProcessedLocation(BaseModel):
    path_train: str
    path_test: str


def main(raw_location: RawLocation, processed_location: ProcessedLocation) -> None:
    get_raw_data(raw_location)
    t_train, t_test = get_train_test_docs(raw_location)
    save_train_test_docs(processed_location, t_train, t_test)


def get_raw_data(raw_location: RawLocation) -> None:
    gdown.download(raw_location.url, raw_location.zip_path, quiet=False)
    with zipfile.ZipFile(raw_location.zip_path, "r") as zip_ref:
        zip_ref.extractall(".")


def get_train_test_docs(raw_location: RawLocation) -> Tuple[str, str]:
    t_train = extract_texts_from_multiple_files(raw_location.path_train)
    t_test = extract_texts_from_multiple_files(raw_location.path_test)
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
    processed_location: ProcessedLocation, t_train: str, t_test: str
) -> None:
    save_data(processed_location.path_train, t_train)
    save_data(processed_location.path_test, t_test)


def save_data(processed_path: str, processed_data: str) -> None:
    with open(processed_path, "w") as f:
        f.write(processed_data)


if __name__ == "__main__":
    raw_location = RawLocation(
        url="https://drive.google.com/uc?id=1jI1cmxqnwsmC-vbl8dNY6b4aNBtBbKy3",
        zip_path="Twitter.zip",
        path_train="Data/train/en",
        path_test="Data/test/en",
    )
    processed_location = ProcessedLocation(
        path_train="Data/train/en.txt",
        path_test="Data/test/en.txt",
    )
    main(raw_location, processed_location)
