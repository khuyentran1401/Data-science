import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from src.process_data import split_data, split_X_y


def test_split_X_y():
    data = pd.DataFrame({"X": [1, 2, 3, 4], "AdoptionSpeed": [1, 2, 3, np.nan]})
    X, y = split_X_y(data)
    assert_series_equal(y, pd.Series([1, 2, 3, np.nan], name="AdoptionSpeed"))


def test_split_data():
    data = pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": [1, 2, 3, 4], "AdoptionSpeed": [1, 2, 3, np.nan]}
    )
    out = split_data.fn(data)
    y_train_out = out["y_train"]
    X_test_out = out["X_test"].reset_index(drop=True)
    y_train_expected = pd.Series([1, 2, 3], name="AdoptionSpeed")
    X_test_expected = pd.DataFrame({"a": [4], "b": [4]})
    assert_series_equal(y_train_out, y_train_expected, check_dtype=False)
    assert_frame_equal(X_test_out, X_test_expected, check_dtype=False)
