#!/usr/bin/env python3
import click
from src.calc import RPNCalculator


@click.command()
@click.argument("expression", nargs=-1, type=str, required=True)
@click.option("-d", "--debug", is_flag=True, help="Enable debug mode.")
def main(expression: list[str], debug: bool):
    expr = " ".join(expression)
    print("[!] Result is: ", RPNCalculator(expr, debug).run())


if __name__ == "__main__":
    main()
