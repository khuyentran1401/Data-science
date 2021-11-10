from hypothesis import given
from hypothesis.strategies import text


def tokenize_sentence(sentence: str):
        return sentence.split(' ')

def join_sentence(sentence: str):
    return ' '.join(sentence)

@given(text())
def test_tokenize_sentence(text: str):

    assert join_sentence(tokenize_sentence(text)) == text