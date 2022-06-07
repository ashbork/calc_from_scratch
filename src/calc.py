import string
import enum
from abc import ABC


class Assoc(enum.Enum):
    Left = 0
    Right = 1


class Token(ABC):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ParenKind(enum.Enum):
    Left = 0
    Right = 1


class ParenToken(Token):
    def __init__(self, kind: ParenKind):
        self.assoc = Assoc.Left
        self.precedence = 0
        self.kind = kind

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ParenToken) and self.kind == other.kind


class NumberToken(Token):
    def __init__(self, val: str) -> None:
        self.val = val

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.val})"


class OpKind(enum.Enum):
    Add = 0
    Sub = 1
    Mul = 2
    Div = 3
    Pow = 4
    UnaryMinus = 5


class OpToken(Token):
    def __init__(self, op: OpKind) -> None:
        self.op = op
        self.precedence = OP_PROPERTIES[self.op][0]
        self.associativity = OP_PROPERTIES[self.op][1]
        self.as_str = OP_PROPERTIES[self.op][2]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.op})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OpToken):
            return self.op == other.op
        return False


OP_PROPERTIES = {
    OpKind.Add: (2, Assoc.Left, "+"),
    OpKind.Sub: (2, Assoc.Left, "-"),
    OpKind.Mul: (3, Assoc.Left, "*"),
    OpKind.Div: (3, Assoc.Left, "/"),
    OpKind.Pow: (4, Assoc.Right, "^"),
    OpKind.UnaryMinus: (5, Assoc.Left, "-"),
}
ALL_OPS = OP_PROPERTIES.keys()
ALL_STR_OPS = [OP_PROPERTIES[op][2] for op in ALL_OPS]

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
        token = self._at_cursor
        match token:
            case "+":
                token = OpToken(OpKind.Add)
            case "-":
                if self.tokens:
                    if isinstance(self.tokens[-1], OpToken):
                        token = OpToken(OpKind.UnaryMinus)
                    else:
                        token = OpToken(OpKind.Sub)
                else:
                    token = OpToken(OpKind.UnaryMinus)
            case "*":
                token = OpToken(OpKind.Mul)
            case "/":
                token = OpToken(OpKind.Div)
            case "^":
                token = OpToken(OpKind.Pow)
            case _:
                raise TokenizerError(f"Expected an operator, got {token}")

        self.cursor += 1
        return token

    def tokenize_paren(self) -> Token:
        """
        Tokenizes a parenthesis.

        Returns:
            Token: A parenthesis token.
        """
        if self._at_cursor == "(":
            token = ParenToken(ParenKind.Left)
        elif self._at_cursor == ")":
            token = ParenToken(ParenKind.Right)
        else:
            raise TokenizerError(f"Expected ( or ), got {self._at_cursor}")
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

            if self._at_cursor in string.digits:
                tokens.append(self.tokenize_num())
            elif self._at_cursor in ALL_OPS:
                tokens.append(self.tokenize_op())
            elif self._at_cursor in "()":
                tokens.append(self.tokenize_paren())
            elif self._at_cursor in string.whitespace:
                self.cursor += 1
            else:
                raise TokenizerError(f"Expected valid token, got {self._at_cursor}")
        # if two tokens of the same type are next to each other, raise an error
        for i in range(len(tokens) - 1):
            if tokens[i] == ParenToken(ParenKind.Right) or tokens[i] == ParenToken(
                ParenKind.Right
            ):
                continue
            elif tokens[i + 1] == ParenToken(ParenKind.Right) or tokens[
                i + 1
            ] == ParenToken(ParenKind.Right):
                continue
            elif isinstance(tokens[i], type(tokens[i + 1])):
                raise TokenizerError(
                    f"Token of type {tokens[i]} followed by another of the same type ({tokens[i+1]}) {tokens}."
                )
        return tokens

    def to_rpn(self, tokens: list[Token]) -> list[Token]:
        """
        Converts a list of tokens to RPN.

        Args:
            tokens (list[Token]): A list of tokens.

        Raises:
            ParserError: Raised on mismatched parentheses.

        Returns:
            list[Token]: A list of tokens in RPN.
        """
        output_q: list[Token] = []
        op_stack: list[ParenToken | OpToken] = []
        for token in tokens:
            match token:
                case ParenToken() as paren:
                    if paren.kind == ParenKind.Left:
                        op_stack.append(paren)
                    elif paren.kind == ParenKind.Right:
                        try:
                            while op_stack[-1] != ParenToken(ParenKind.Left):
                                output_q.append(op_stack.pop(-1))
                        except IndexError:
                            raise ParserError("mismatched parentheses")
                        if op_stack[-1] != ParenToken(ParenKind.Left):
                            raise ParserError("mismatched parentheses")
                        op_stack.pop(-1)

                case OpToken() as op:
                    print(op_stack)

                    while op_stack and (
                        op_stack[-1] != ParenToken(ParenKind.Left)
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

                case NumberToken() as num:
                    output_q.append(num)
                case _:
                    raise AssertionError("unreachable")
        while op_stack:
            if op_stack[-1] == ParenToken(ParenKind.Left):
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
                        if op.op == OpKind.Add:
                            stack.append(stack.pop(-1) + stack.pop(-1))
                        elif op.op == OpKind.Sub:
                            stack.append(-stack.pop(-1) + stack.pop(-1))
                        elif op.op == OpKind.Mul:
                            stack.append(stack.pop(-1) * stack.pop(-1))
                        elif op.op == OpKind.Div:
                            stack.append(1 / stack.pop(-1) * stack.pop(-1))
                        elif op.op == OpKind.Pow:
                            exp = stack.pop(-1)
                            stack.append(stack.pop(-1) ** exp)
                        elif op.op == OpKind.UnaryMinus:
                            stack.append(-stack.pop(-1))
                        else:
                            raise AssertionError("should be unreachable")
                    case NumberToken() as num:
                        stack.append(float(num.val))
                    case _:
                        raise AssertionError("unreachable")
            except IndexError:
                raise ParserError("too many operators")
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
