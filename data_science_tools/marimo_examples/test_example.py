import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pytest

    return (pytest,)


@app.function
def add_numbers(a, b):
    """Return the sum of two numbers."""
    return a + b


@app.function
def test_add_numbers():
    assert add_numbers(2, 3) == 5


@app.cell
def _(pytest):
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (2, 3, 5),
            (-1, 1, 0),
            (0, 0, 0),
        ],
    )
    def test_multiple_add_numbers(a, b, expected):
        assert add_numbers(a, b) == expected

    return


if __name__ == "__main__":
    app.run()
