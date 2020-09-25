import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from src.process import extract_sentiment
import pytest


def test_extract_sentiment():

    text = 'Today I found a duck and I am happy'

    sentiment = extract_sentiment(text)

    assert sentiment > 0