import math
import string
import enum
from abc import ABC


class Assoc(enum.Enum):
    Left = 0
    Right = 1


class Token(ABC):
    def __init__(self):
        self.precedence = 0
        self.associativity = Assoc.Left

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class NumberToken(Token):
    def __init__(self, val: str) -> None:
        self.val = val

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.val})"


class ConstantKind(enum.Enum):
    PI = 0
    E = 1


class ConstantToken(Token):
    kind_to_value = {
        ConstantKind.PI: math.pi,
        ConstantKind.E: math.e,
    }

    def __init__(self, kind: ConstantKind) -> None:
        self.kind = kind
        self.value = ConstantToken.kind_to_value[kind]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.kind})"


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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.kind})"


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


class FuncKind(enum.Enum):
    Sin = 0
    Cos = 1
    Tan = 2
    Log = 3
    Ln = 4
    Exp = 5
    Abs = 6
    Sqrt = 7


class FuncToken(Token):
    def __init__(self, kind: FuncKind) -> None:
        self.kind = kind
        super().__init__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.kind})"


class TokenizerError(Exception):
    pass


class ParserError(Exception):
    pass


class RPNCalculator:
    def __init__(self, to_parse: str, debug: bool = False) -> None:
        self.cursor = 0
        self.input = to_parse.strip()
        self.debug = debug

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

    def _peek_prev(self) -> str | None:
        if self.cursor - 1 < 0:
            return None
        return self.input[self.cursor - 1]

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
                        f"Expected one or less .|, symbols in number, got {val}"
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
                    if isinstance(self.tokens[-1], OpToken) or isinstance(
                        self.tokens[-1], FuncToken
                    ):
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

    def tokenize_constant(self) -> ConstantToken:
        val = ""
        while (not self._done()) and self._at_cursor in string.ascii_uppercase:
            val += self._at_cursor
            self.cursor += 1
        match val:
            case "PI":
                token = ConstantToken(ConstantKind.PI)
            case "E":
                token = ConstantToken(ConstantKind.E)
            case _:
                raise TokenizerError(f"Expected mathematical constant, got {val}")
        return token

    def tokenize_func(self) -> FuncToken:
        val = ""
        while (not self._done()) and self._at_cursor in string.ascii_letters:
            val += self._at_cursor
            self.cursor += 1
        match val:
            case "sin":
                token = FuncToken(FuncKind.Sin)
            case "cos":
                token = FuncToken(FuncKind.Cos)
            case "tan":
                token = FuncToken(FuncKind.Tan)
            case "log":
                token = FuncToken(FuncKind.Log)
            case "ln":
                token = FuncToken(FuncKind.Ln)
            case "exp":
                token = FuncToken(FuncKind.Exp)
            case "abs":
                token = FuncToken(FuncKind.Abs)
            case "sqrt":
                token = FuncToken(FuncKind.Sqrt)
            case _:
                raise TokenizerError(f"Expected a function, got {val}")
        return token

    def tokenize(self) -> list[Token]:
        """
        Tokenizes the input string.

        Returns:
            list[Token]: A list of tokens.
        """
        self.tokens: list[Token] = []
        while not self._done():
            if self._at_cursor in string.digits:
                self.tokens.append(self.tokenize_num())
            elif self._at_cursor in ALL_STR_OPS:
                self.tokens.append(self.tokenize_op())
            elif self._at_cursor in "()":
                self.tokens.append(self.tokenize_paren())
            elif self._at_cursor in string.ascii_uppercase:
                self.tokens.append(self.tokenize_constant())
            elif self._at_cursor in string.ascii_lowercase:
                self.tokens.append(self.tokenize_func())
            elif self._at_cursor in string.whitespace:
                self.cursor += 1
            else:
                raise TokenizerError(f"Expected valid token, got {self._at_cursor}")
        if self.debug:
            print("[1] Tokenizing into: ", self.tokens)
        return self.tokens

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
        op_stack: list[ParenToken | OpToken | FuncToken] = []
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
                        if op_stack:
                            if isinstance(op_stack[-1], FuncToken):
                                output_q.append(op_stack.pop(-1))

                case OpToken() as op:
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

                case ConstantToken() as const:
                    output_q.append(const)

                case FuncToken() as func:
                    op_stack.append(func)
                case _:
                    raise AssertionError("unreachable")
        while op_stack:
            if op_stack[-1] == ParenToken(ParenKind.Left):
                raise ParserError("mismatched parentheses")
            output_q.append(op_stack.pop(-1))
        if self.debug:
            print("[2] Converting to RPN: ", output_q)
        return output_q

    def calculate(self, rpn: list[Token]) -> float:
        """
        Iterates through the RPN list, and for each token, if it's an operation it performs the operation on the last two numbers in the stack.
        or if it's a number, it pushes it to the stack.
        """
        if self.debug:
            print("[3] Calculating: ", rpn)
        stack: list[float] = []
        for token in rpn:
            try:
                match token:
                    case OpToken() as op:
                        if self.debug:
                            print("[+] Applying operation: ", op.op.name)
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
                        if self.debug:
                            print("[+] Adding number to stack: ", num)
                        stack.append(float(num.val))
                    case FuncToken() as func:
                        if self.debug:
                            print("[+] Applying function ", func.kind.name)
                        if func.kind == FuncKind.Sin:
                            stack.append(round(math.sin(stack.pop(-1))))
                        elif func.kind == FuncKind.Cos:
                            stack.append(round(math.cos(stack.pop(-1))))
                        elif func.kind == FuncKind.Tan:
                            stack.append(math.tan(stack.pop(-1)))
                        elif func.kind == FuncKind.Ln:
                            stack.append(math.log(stack.pop(-1)))
                        elif func.kind == FuncKind.Log:
                            stack.append(math.log(stack.pop(-1), 10))
                        elif func.kind == FuncKind.Exp:
                            stack.append(math.exp(stack.pop(-1)))
                        elif func.kind == FuncKind.Abs:
                            stack.append(math.fabs(stack.pop(-1)))
                        elif func.kind == FuncKind.Sqrt:
                            stack.append(math.sqrt(stack.pop(-1)))
                        else:
                            raise AssertionError("should be unreachable")
                    case ConstantToken() as const:
                        if self.debug:
                            print("[+] Adding constant to stack: ", const, const.value)
                        stack.append(const.value)
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
