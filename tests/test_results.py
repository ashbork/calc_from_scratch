import pytest
from src.calc import RPNCalculator
import src.calc


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("1 + 4", 5),
        ("1 + 4 * 2", 9),
        ("3 * (4 + 2)", 18),
        ("0.5 * (1 + 2)", 1.5),
    ],
)
def test_calculate(expression: str, expected: float):
    assert RPNCalculator(expression).run() == expected


def test_calculate_error():
    with pytest.raises(src.calc.ParserError):
        RPNCalculator("1 + 4 *").run()
