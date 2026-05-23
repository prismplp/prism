import dataclasses
from functools import total_ordering
from typing import Any, Literal, TypeAlias, Tuple, Optional, List

import torch

import tprism.parser

ParsedTerm: TypeAlias = str | int | dict[str, Any]
IndexType: TypeAlias = Literal["symbol", "range", "index"]

@total_ordering
@dataclasses.dataclass
class TensorIndexRef:
    """Normalized representation of one tensor index expression."""

    index_type: IndexType
    start: int = -1
    end: int = -1
    step: int = 1
    symbol: str = ""

    # for sorting and comparing TensorIndexRef objects based on their symbol
    def __lt__(self, other: "TensorIndexRef") -> bool:
        if not isinstance(other, TensorIndexRef):
            return NotImplemented
        return self.symbol < other.symbol
    
    def __repr__(self) -> str:
        if self.index_type == "symbol" or self.index_type == "range":
            if self.start == 0 and self.end == -1 and self.step == 1:
                return f"{self.symbol!r}"
            elif self.start == 0 and self.end == -1:
                return f"{self.symbol!r}@[::{self.step}]"
            elif self.step == 1:
                return f"{self.symbol!r}@[{self.start}:{self.end}]"
            else:
                return f"{self.symbol!r}@[{self.start}:{self.end}:{self.step}]"
        elif self.index_type == "index":
            return f"index[{self.start}]"
        return ""
    
def parse_el(element: ParsedTerm, default: int = 0) -> int:
    """Parse one start, end, or step element of a tensor range expression."""
    if isinstance(element, str):
        if element in {"'-'", "-"}:
            return default
    elif isinstance(element, dict) and "nullary" in element:
        return default
    elif isinstance(element, dict) and "unary" in element:
        expr = element.get("expr")
        if not isinstance(expr, int):
            raise ValueError(f"Expected integer unary expression: {element!r}")
        if element["unary"] == "-":
            return -expr
        if element["unary"] == "+":
            return expr
    elif isinstance(element, int):
        return element

    raise ValueError(
        "Unparsable object at the start/end/step of the range object: "
        f"{element!r} ({type(element)!r})"
    )


def carcdr_tolist(obj: ParsedTerm) -> list[ParsedTerm]:
    """Flatten a right-nested Prism comma term into a Python list."""
    current_obj = obj
    out_list: list[ParsedTerm] = []

    while True:
        if not isinstance(current_obj, dict):
            out_list.append(current_obj)
            break
        if current_obj.get("name") == "','":
            out_list.append(current_obj["args"][0])
            current_obj = current_obj["args"][1]
        else:
            break

    return out_list


def _parse_range(arg_list: list[ParsedTerm]) -> TensorIndexRef:
    """Build a range index reference from Prism range arguments."""
    if not arg_list:
        raise ValueError("Range index expression requires at least one argument")

    start = 0
    end = -1
    step = 1
    symbol = tprism.parser.serialize_term(arg_list[0])

    if len(arg_list) >= 2:
        start = parse_el(arg_list[1], 0)
    if len(arg_list) >= 3:
        end = parse_el(arg_list[2], -1)
    if len(arg_list) >= 4:
        step = parse_el(arg_list[3], 1)

    return TensorIndexRef("range", start, end, step, symbol)


def parse_tensor_index(index_expr: str) -> TensorIndexRef:
    """Parse a Prism tensor index expression into a TensorIndexRef."""
    parsed = tprism.parser.parse_term(index_expr)

    if isinstance(parsed, str):
        return TensorIndexRef("symbol", 0, -1, 1, index_expr)
    if isinstance(parsed, int):
        return TensorIndexRef("index", int(parsed), int(parsed), 1, index_expr)
    if not isinstance(parsed, dict):
        raise ValueError(f"Unsupported tensor index expression: {index_expr!r}")
    if parsed.get("name") == "r":
        return _parse_range(parsed["args"])
    if parsed.get("name") == "','":
        return _parse_range(carcdr_tolist(parsed))

    raise ValueError(f"Unsupported tensor index expression: {index_expr!r}")

def extract_tensor_shape(shape: Tuple[Optional[int], ...], index_ref_list: list[TensorIndexRef]) -> Tuple[Optional[int], ...]:
    """Extract the shape of a tensor after applying index references."""
    new_shape: List[Optional[int]] = []
    for i, (dim_size, index_ref) in enumerate(zip(shape, index_ref_list)):
        if dim_size is not None:
            if index_ref.index_type == "symbol":
                new_shape.append(dim_size)
            elif index_ref.index_type == "range":
                if index_ref.start<0:
                    start=dim_size+index_ref.start
                else:
                    start=index_ref.start
                if index_ref.end<0:
                    end=dim_size+index_ref.end
                else:
                    end=index_ref.end
                new_dim=len(range(start, end, index_ref.step))
                if new_dim > 0:
                    new_shape.append(new_dim)
            elif index_ref.index_type == "index":
                #new_shape.append(1) # without keepdims
                pass
        else:
            new_shape.append(None)
    return tuple(new_shape)

def extract_tensor(
    tensor: torch.Tensor, index_ref_list: list[TensorIndexRef]
) -> tuple[torch.Tensor, tuple[int | slice, ...], list[str]]:
    """Apply tensor index references and return the indexed tensor and symbols."""
    indices: list[int | slice] = []
    out_symbols: list[str] = []

    for index_ref in index_ref_list:
        if index_ref.index_type == "symbol":
            indices.append(slice(None))
            out_symbols.append(index_ref.symbol)
        elif index_ref.index_type == "range":
            indices.append(slice(index_ref.start, index_ref.end, index_ref.step))
            out_symbols.append(index_ref.symbol)
        elif index_ref.index_type == "index":
            indices.append(index_ref.start)

    index_tuple = tuple(indices)
    return tensor[index_tuple], index_tuple, out_symbols


def main() -> None:
    """Run a small manual parser and tensor extraction example."""
    s1 = "i"
    s2 = "r(i,  10, 20  )"
    s3 = "r(j,'-' , 20 )"
    s4 = "','(k, 15 , 15 )"
    s5 = "30"
    s6 = "r(l, 10 ,- )"
    s7 = "r(f(a,b), 10 ,- )"
    s8 = "','(k,','(10,20))"
    s9 = "','(k,10)"
    s10 = "','(j,','(-,','(10,','(1,','(2,','(3,','(a,','(b,c))))))))"
    s11 = "','(f(1,a,b),','(-,','(10,','(-5,','(h(-,a),g(c))))))"

    for index_expr in [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]:
        parsed_index = parse_tensor_index(index_expr)
        print(parsed_index)

    tensor = torch.zeros((100, 100, 100))
    extracted, indices, symbols = extract_tensor(
        tensor, [parse_tensor_index(index_expr) for index_expr in [s1, s2, s4]]
    )
    print("X=", tensor.shape)
    print("A=", extracted.shape)
    print("Index=", indices)
    print("Index symbol=", symbols)
    
    new_s=extract_tensor_shape(tensor.shape, [parse_tensor_index(index_expr) for index_expr in [s1, s2, s4]])
    print("shape=", new_s)

if __name__ == "__main__":
    main()
