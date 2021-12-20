from sklearn.preprocessing import LabelEncoder
from hypothesis import given, strategies as st
from hypothesis.strategies import integers, floats, lists, text, characters
from hypothesis.extra.numpy import arrays, unicode_string_dtypes
import numpy as np   
from numpy.testing import assert_equal   

@given(arrays(unicode_string_dtypes(min_len=2), shape=(10)))
def test_encoder(labels):
    le = LabelEncoder()
    le.fit(labels)
    encoded_labels = le.transform(labels)
    assert assert_equal(labels, le.inverse_transform(encoded_labels))

print(arrays(str, shape=(10)).example())