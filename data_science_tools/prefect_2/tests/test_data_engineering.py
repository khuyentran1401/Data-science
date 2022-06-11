import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from src.data_engineering import *


@pytest.fixture
def processed():
    return {
        "breeds.primary": ["Domestic Short Hair", "Domestic Long Hair"],
        "age": ["Adult", "Adult"],
    }


def test_process_json():
    data = [
        {
            "breeds": {"primary": "Domestic Short Hair", "secondary": None},
            "age": "Adult",
        },
        {
            "breeds": {"primary": "Domestic Long Hair", "secondary": None},
            "age": "Adult",
        },
    ]
    attributes = ["breeds.primary", "age"]
    processed = process_json.fn(data, attributes)
    assert processed == processed


def test_convert_to_dataframe(processed):
    assert_frame_equal(pd.DataFrame(processed), convert_to_dataframe.fn(processed))
