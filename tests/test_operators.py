import pytest
from src.calc import RPNCalculator


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("-2", -2),
        ("-2.5", -2.5),
        ("---2", -2),
        ("-2 * 9", -18),
        ("-2 * -9", 18),
        ("-2 * -9 * -2", -36),
        ("-(10*5)", -50),
    ],
)
def test_unary(expression: str, expected: float):
    assert RPNCalculator(expression).run() == expected
