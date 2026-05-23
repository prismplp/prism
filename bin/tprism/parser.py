import re
from collections.abc import Iterable, Iterator
from typing import Any, TypeAlias

Token: TypeAlias = tuple[str, str]
Term: TypeAlias = Any
OutputPair: TypeAlias = tuple[str, str]
SwitchEntry: TypeAlias = dict[str, Any]
SwitchRow: TypeAlias = tuple[list[Any], list[str]]

# Priority: higher values bind more strongly.
PRECEDENCE: dict[str, int] = {
    ":-": 0,
    "?-": 0,
    ";": 100,
    ",": 200,
    "=": 500,
    "<": 500,
    ">": 500,
    "=<": 500,
    ">=": 500,
    "==": 500,
    "=\\=": 500,
    "=:=": 500,
    "\\==": 500,
    "+": 700,
    "-": 700,
    "*": 800,
    "/": 800,
}
PREFIX_OPS = {"+", "-", ":-"}
NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")


def tokenize(source: str) -> Iterator[Token]:
    """Yield parser tokens from a Prism-like term string."""
    token_specification = [
        ("STRING", r"\"(\\.|[^\"\\])*\"|\'(\\.|[^\'\\])*\'"),
        ("OP", r"[\+\-\*/=><:]+"),
        ("NUMBER", r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][\+\-]?\d+)?"),
        ("NAME", r"[A-Za-z_][A-Za-z0-9_]*"),
        ("COMMA", r","),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACK", r"\["),
        ("RBRACK", r"\]"),
        ("SKIP", r"\s+"),
        ("OTHER", r" "),
    ]
    token_regex = "|".join(f"(?P<{name}>{regex})" for name, regex in token_specification)

    for match in re.finditer(token_regex, source):
        kind = match.lastgroup
        value = match.group()
        if kind == "SKIP":
            continue
        if kind == "OTHER":
            raise SyntaxError(f"Unexpected character: {value}")
        if kind is None:
            raise SyntaxError(f"Unexpected token: {value}")
        yield kind, value


class TokenStream:
    """Small cursor wrapper around a token sequence."""

    def __init__(self, tokens: Iterable[Token]) -> None:
        self.tokens = list(tokens)
        self.pos = 0

    def peek(self) -> Token | None:
        """Return the current token without consuming it."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> Token | None:
        """Consume and return the current token."""
        token = self.peek()
        self.pos += 1
        return token


def parse_tuple_or_paren_expr(token_stream: TokenStream) -> Term:
    """Parse a parenthesized expression or tuple expression."""
    items: list[Term] = [parse_expr(token_stream)]

    while True:
        token = token_stream.peek()
        if token is None:
            raise SyntaxError("Unclosed ')'")
        if token[0] == "COMMA":
            token_stream.next()
            items.append(parse_expr(token_stream))
        elif token[0] == "RPAREN":
            token_stream.next()
            break
        else:
            raise SyntaxError(f"Unexpected token in tuple or paren: {token}")

    if len(items) == 1:
        return items[0]
    return {"tuple": items}


def parse_expr(token_stream: TokenStream, min_prec: int = 0) -> Term:
    """Parse an expression using prefix and binary operator precedence."""
    token = token_stream.peek()
    if token is None:
        raise SyntaxError("Unexpected end of input")

    kind, value = token
    if kind == "OP" and value in PREFIX_OPS:
        token_stream.next()
        next_token = token_stream.peek()
        if next_token is None or next_token[0] in {"COMMA", "RPAREN", "RBRACK"}:
            return {"nullary": value}
        node: Term = {"unary": value, "expr": parse_expr(token_stream, PRECEDENCE[value])}
    else:
        node = parse_atom(token_stream)

    while True:
        token = token_stream.peek()
        if token is None or token[0] != "OP":
            break
        op = token[1]
        precedence = PRECEDENCE.get(op)
        if precedence is None or precedence < min_prec:
            break
        token_stream.next()
        rhs = parse_expr(token_stream, precedence + 1)
        node = {"binop": op, "left": node, "right": rhs}

    return node


def parse_atom(token_stream: TokenStream) -> Term:
    """Parse a name, string, number, function call, list, or parenthesized term."""
    token = token_stream.peek()
    if token is None:
        raise SyntaxError("Unexpected end of input")

    kind, value = token
    if kind in {"NAME", "STRING"}:
        token_stream.next()
        next_token = token_stream.peek()
        if next_token and next_token[0] == "LPAREN":
            token_stream.next()
            args = parse_args(token_stream, end_kind="RPAREN")
            return {"name": value, "args": args}
        return value

    if kind == "NUMBER":
        token_stream.next()
        return float(value) if "." in value or "e" in value.lower() else int(value)

    if kind == "LPAREN":
        token_stream.next()
        return parse_tuple_or_paren_expr(token_stream)

    if kind == "LBRACK":
        token_stream.next()
        return parse_args(token_stream, end_kind="RBRACK")

    raise SyntaxError(f"Unexpected token: {token}")


def parse_args(token_stream: TokenStream, end_kind: str) -> list[Term]:
    """Parse comma-separated arguments up to the given closing token kind."""
    items: list[Term] = []

    while True:
        token = token_stream.peek()
        if token is None:
            raise SyntaxError(f"Unclosed {end_kind}")
        if token[0] == end_kind:
            token_stream.next()
            break
        if token[0] == "COMMA":
            token_stream.next()
            continue
        items.append(parse_expr(token_stream))

    return items


def parse_term(source: str) -> Term:
    """Parse a Prism-like term string into nested Python values."""
    tokens = TokenStream(tokenize(source))
    return parse_expr(tokens)


def parse_output_(obj: Term) -> OutputPair | None:
    """Parse one output assignment term into a serialized key-value pair."""
    if isinstance(obj, dict) and obj.get("binop") == "=":
        lhs = serialize_term(obj["left"])
        rhs = serialize_term(obj["right"])
        return lhs, rhs
    return None


def parse_output(source: str) -> list[OutputPair | None] | None:
    """Parse comma-separated output assignments."""
    obj = parse_term(f"({source})")
    if isinstance(obj, dict) and "tuple" in obj:
        return [parse_output_(item) for item in obj["tuple"]]
    if isinstance(obj, dict) and "binop" in obj:
        return [parse_output_(obj)]
    return None


def serialize_term(
    obj: Term, unary_op_paren: bool = True, binary_op_paren: bool = True
) -> str:
    """Serialize a parsed term back to a Prism-like term string."""
    if isinstance(obj, dict) and "name" in obj:
        args = ",".join(serialize_term(arg) for arg in obj["args"])
        return f"{obj['name']}({args})"
    if isinstance(obj, dict) and "nullary" in obj:
        return str(obj["nullary"])
    if isinstance(obj, dict) and "unary" in obj:
        expr = serialize_term(obj["expr"])
        value = f"{obj['unary']}{expr}"
        return f"({value})" if unary_op_paren else value
    if isinstance(obj, dict) and "binop" in obj:
        lhs = serialize_term(obj["left"])
        rhs = serialize_term(obj["right"])
        value = f"{lhs}{obj['binop']}{rhs}"
        return f"({value})" if binary_op_paren else value
    if isinstance(obj, dict) and "tuple" in obj:
        return "(" + ",".join(serialize_term(item) for item in obj["tuple"]) + ")"
    if isinstance(obj, list):
        return "[" + ",".join(serialize_term(item) for item in obj) + "]"
    if isinstance(obj, str):
        if NAME_PATTERN.match(obj):
            return obj
        return '"' + obj.replace('"', '\\"') + '"'
    if isinstance(obj, int | float):
        return str(obj)

    raise TypeError(f"Unsupported type: {type(obj)}")


def read_sw(filename: str) -> list[SwitchEntry]:
    """Read switch declarations from a file."""
    sw_list: list[SwitchEntry] = []

    with open(filename, encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if not stripped:
                continue
            sw = parse_term(stripped[:-1])
            if sw["name"] != "switch":
                continue

            term_obj = sw["args"][0]
            sw_list.append(
                {
                    "term": serialize_term(term_obj),
                    "term_obj": term_obj,
                    "status": sw["args"][1],
                    "values": sw["args"][2],
                    "params": sw["args"][3],
                }
            )

    return sw_list


def read_sw_data(filename: str, use_array: bool = False) -> tuple[list[SwitchRow], int]:
    """Convert switch declarations into table rows and a maximum arity."""
    sw_list = read_sw(filename)
    data: list[SwitchRow] = []

    for element in sw_list:
        term_obj = element["term_obj"]
        name = term_obj["name"] if "name" in term_obj else str(term_obj)
        arity = len(term_obj["args"]) if "name" in term_obj else 0
        args = [serialize_term(arg) for arg in term_obj["args"]] if "name" in term_obj else []

        if use_array:
            line = [
                name,
                arity,
                element["term"],
                element["status"],
                element["values"],
                element["params"],
            ]
        else:
            values = "[" + ",".join(map(str, element["values"])) + "]"
            params = "[" + ",".join(map(str, element["params"])) + "]"
            line = [name, arity, element["term"], element["status"], values, params]

        data.append((line, args))

    max_args = max(max((len(args) for _, args in data), default=0), 5)
    return data, max_args


def sw2tsv(filename: str, out_filename: str) -> None:
    """Write switch declarations as a tab-separated values file."""
    data, max_args = read_sw_data(filename)

    with open(out_filename, "w", encoding="utf-8") as output_file:
        header = ["Name", "Arity", "Term", "Status", "Vals", "Param"]
        arg_headers = [f"Arg{i + 1}" for i in range(max_args)]
        output_file.write("\t".join(header + arg_headers))
        output_file.write("\n")

        for line, args in data:
            row = line + args + [""] * (max_args - len(args))
            output_file.write("\t".join(map(str, row)))
            output_file.write("\n")
