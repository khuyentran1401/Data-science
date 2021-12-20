from hypothesis import given, strategies as st
from hypothesis.strategies import integers, floats
from statistics import mean


@given(st.lists(st.floats(allow_infinity=False, allow_nan=False), min_size=1))
def test_mean_is_in_bounds(ls):
  assert min(ls) <= mean(ls) <= max(ls)

@given(floats(), floats())
def test_floats_are_commutative(x, y):
    assert x + y == y + x