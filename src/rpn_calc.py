import string
import enum
from abc import ABC


class Assoc(enum.Enum):
    Left = 0
    Right = 1


class Token(ABC):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class NumberToken(Token):
    def __init__(self, val: str) -> None:
        self.val = val

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.val})"


class OpToken(Token):
    def __init__(self, op: str) -> None:
        self.op = op
        self.precedence = OP_PROPERTIES[self.op][0]
        self.associativity = OP_PROPERTIES[self.op][1]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.op})"


OP_PROPERTIES = {
    "*": (3, Assoc.Left),
    "/": (3, Assoc.Left),
    "+": (2, Assoc.Left),
    "-": (2, Assoc.Left),
    "^": (4, Assoc.Right),
    "(": (0, Assoc.Left),
    ")": (0, Assoc.Left),
}
ALL_OPS = OP_PROPERTIES.keys()

ALLOWED_FOR_NUMS = string.digits + ",."


class TokenizerError(Exception):
    pass


class ParserError(Exception):
    pass


class RPNCalculator:
    def __init__(self, to_parse: str) -> None:
        self.cursor = 0
        self.input = to_parse.strip()

    @property
    def _at_cursor(self) -> str:
        """
        Helper property, returns the character at the cursor position.
        """
        return self.input[self.cursor]

    def _done(self) -> bool:
        """
        Returns True if we're at the end of input, False otherwise.
        """
        return self.cursor == len(self.input)

    def _peek(self) -> None | str:
        """
        Returns the next character if not at the end of input.
        """
        return None if self._done() else self.input[self.cursor + 1]

    def tokenize_num(self) -> Token:
        """
        Tokenizes a number, looking ahead until it reaches its end, at which point it returns a
        NumberToken.

        Raises:
            TokenizerError: Raised when the number has too many periods/commas.

        Returns:
            Token: Constructed token
        """
        val = ""
        point = False
        while (not self._done()) and self._at_cursor in ALLOWED_FOR_NUMS:
            if self._at_cursor in ",.":
                if point:
                    raise TokenizerError(
                        "Expected one or less .|, symbols in number, got more"
                    )
                point = True
            val += self._at_cursor
            self.cursor += 1
        val.strip(".,")
        return NumberToken(val)

    def tokenize_op(self) -> Token:
        """
        Tokenizes an operation.

        Returns:
            Token: an operation
        """
        token = OpToken(self.input[self.cursor])
        self.cursor += 1
        return token

    def tokenize(self) -> list[Token]:
        """
        Tokenizes the input string.

        Returns:
            list[Token]: A list of tokens.
        """
        tokens: list[Token] = []
        while not self._done():
            current_char = self.input[self.cursor]
            if current_char in string.digits:
                tokens.append(self.tokenize_num())
            elif current_char in ALL_OPS:
                tokens.append(self.tokenize_op())
            elif current_char in string.whitespace:
                self.cursor += 1
            else:
                raise TokenizerError(f"Expected valid token, got {current_char}")
        # if two tokens of the same type are next to each other, raise an error
        for i in range(len(tokens)):
            if isinstance(tokens[i], type(tokens[i + 1])):
                raise TokenizerError(
                    f"Token of type {tokens[i]} followed by another of the same type."
                )
        return tokens

    def to_rpn(self, tokens: list[Token]) -> list[Token]:
        output_q: list[Token] = []
        op_stack: list[OpToken] = []
        for token in tokens:
            match token:
                case OpToken() as op:
                    print(op_stack)
                    if op.op == "(":
                        op_stack.append(op)
                    elif op.op == ")":
                        try:
                            while op_stack[-1].op != "(":
                                output_q.append(op_stack.pop(-1))
                        except IndexError:
                            raise ParserError("mismatched parentheses")
                        if op_stack[-1].op != "(":
                            raise ParserError("mismatched parentheses")
                        op_stack.pop(-1)

                    else:
                        while op_stack and (
                            op_stack[-1] is not OpToken("(")
                            and (
                                op_stack[-1].precedence > op.precedence
                                or (
                                    op_stack[-1].precedence == op
                                    and op.associativity == Assoc.Left
                                )
                            )
                        ):
                            output_q.append(op_stack.pop(-1))
                        op_stack.append(op)
                    pass
                case NumberToken() as num:
                    output_q.append(num)
                case _:
                    raise AssertionError("unreachable")
        while op_stack:
            if op_stack[-1] is OpToken("("):
                raise ParserError("mismatched parentheses")
            output_q.append(op_stack.pop(-1))
        print(output_q)
        return output_q

    def calculate(self, rpn: list[Token]) -> float:
        """
        Iterates through the RPN list, and for each token, if it's an operation it performs the operation on the last two numbers in the stack.
        or if it's a number, it pushes it to the stack.
        """

        stack: list[float] = []
        for token in rpn:
            try:
                match token:
                    case OpToken() as op:
                        if op.op == "+":
                            stack.append(stack.pop(-1) + stack.pop(-1))
                        elif op.op == "-":
                            stack.append(-stack.pop(-1) + stack.pop(-1))
                        elif op.op == "*":
                            stack.append(stack.pop(-1) * stack.pop(-1))
                        elif op.op == "/":
                            stack.append(1 / stack.pop(-1) * stack.pop(-1))
                        elif op.op == "^":
                            exp = stack.pop(-1)
                            stack.append(stack.pop(-1) ** exp)
                        else:
                            raise AssertionError("unreachable")
                    case NumberToken() as num:
                        stack.append(float(num.val))
                    case _:
                        raise AssertionError("unreachable")
            except IndexError:
                raise ParserError("stack underflow - too many operators")
        return stack[-1]

    def run(self) -> float:
        """
        Runs the calculator.

        Returns:
            float: The result of the calculation.
        """
        tokens = self.tokenize()
        rpn = self.to_rpn(tokens)
        return self.calculate(rpn)
