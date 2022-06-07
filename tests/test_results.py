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
        ("5^2", 25),
        ("5^2^3", 5 ** (2**3)),
        (
            "2^(2.5-1)",
            2 ** (2.5 - 1),
        ),
    ],
)
def test_calculate(expression: str, expected: float):
    assert RPNCalculator(expression).run() == expected


# Test for the error handling of the calculator.
def test_calculate_error():
    with pytest.raises(src.calc.ParserError):
        RPNCalculator("1 + 4 *").run()


def test_calculate_error_2():
    with pytest.raises(src.calc.ParserError):
        RPNCalculator("1 (").run()


def test_calculate_error_3():
    with pytest.raises(src.calc.ParserError):
        RPNCalculator("1 + 4 * 2 +").run()


def test_calculate_error_4():
    with pytest.raises(src.calc.ParserError):
        RPNCalculator("())+").run()


def test_calculate_error_5():
    with pytest.raises(src.calc.ParserError):
        RPNCalculator("())))))+").run()


def test_point_error():
    with pytest.raises(src.calc.TokenizerError):
        RPNCalculator("1.2.3").run()
