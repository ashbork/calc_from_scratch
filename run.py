#!/usr/bin/env python3
import click
from src.rpn_calc import RPNCalculator


@click.command()
@click.argument("expression", nargs=-1, type=str, required=True)
def main(expression: list[str]):
    expr = " ".join(expression)
    print(RPNCalculator(expr).run())


if __name__ == "__main__":
    main()
