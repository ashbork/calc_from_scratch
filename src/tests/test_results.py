import pytest
from calc import RPNCalculator


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("1 + 4", 5),
        ("1 + 4 * 2", 9),
        ("1 + 4 * 2 + 3", 12),
        ("1 + 4 * 2 + 3 - 1", 11),
        ("1 + 4 * 2 + 3 - 1 / 2", 10),
        ("1 + 4 * 2 + 3 - 1 / 2 ^ 2", 9),
        ("1 + 4 * 2 + 3 - 1 / 2 ^ 2 ^ 3", 8),
        ("1 + 4 * 2 + 3 - 1 / 2 ^ 2 ^ 3 ^ 4", 7),
    ],
)
def test_calculate(expression: str, expected: float):
    assert RPNCalculator(expression).run() == expected
