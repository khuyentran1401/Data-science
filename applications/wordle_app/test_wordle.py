import pytest
from wordle import *


@pytest.mark.parametrize(
    "test_input,expected", [("hello", True), ("abc", False), ("appl", False)]
)
def test_is_word(test_input, expected):
    assert is_word(test_input) == expected


@pytest.mark.repeat(100)
def test_get_today_word():
    assert len(get_today_word()) == 5
