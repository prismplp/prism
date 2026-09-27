from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor
Operator: SwType
Probabilistic: SwType
Tensor: SwType

class DataRecord(_message.Message):
    __slots__ = ["items"]
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, items: _Optional[_Iterable[str]] = ...) -> None: ...

class ExplGraph(_message.Message):
    __slots__ = ["goals", "root_list"]
    GOALS_FIELD_NUMBER: _ClassVar[int]
    ROOT_LIST_FIELD_NUMBER: _ClassVar[int]
    goals: _containers.RepeatedCompositeFieldContainer[ExplGraphGoal]
    root_list: _containers.RepeatedCompositeFieldContainer[RankRoot]
    def __init__(self, goals: _Optional[_Iterable[_Union[ExplGraphGoal, _Mapping]]] = ..., root_list: _Optional[_Iterable[_Union[RankRoot, _Mapping]]] = ...) -> None: ...

class ExplGraphGoal(_message.Message):
    __slots__ = ["node", "paths"]
    NODE_FIELD_NUMBER: _ClassVar[int]
    PATHS_FIELD_NUMBER: _ClassVar[int]
    node: ExplGraphNode
    paths: _containers.RepeatedCompositeFieldContainer[ExplGraphPath]
    def __init__(self, node: _Optional[_Union[ExplGraphNode, _Mapping]] = ..., paths: _Optional[_Iterable[_Union[ExplGraphPath, _Mapping]]] = ...) -> None: ...

class ExplGraphNode(_message.Message):
    __slots__ = ["goal", "id", "sorted_id"]
    GOAL_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    SORTED_ID_FIELD_NUMBER: _ClassVar[int]
    goal: GoalTerm
    id: int
    sorted_id: int
    def __init__(self, id: _Optional[int] = ..., sorted_id: _Optional[int] = ..., goal: _Optional[_Union[GoalTerm, _Mapping]] = ...) -> None: ...

class ExplGraphPath(_message.Message):
    __slots__ = ["nodes", "operators", "prob_switches", "tensor_switches"]
    NODES_FIELD_NUMBER: _ClassVar[int]
    OPERATORS_FIELD_NUMBER: _ClassVar[int]
    PROB_SWITCHES_FIELD_NUMBER: _ClassVar[int]
    TENSOR_SWITCHES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[ExplGraphNode]
    operators: _containers.RepeatedCompositeFieldContainer[SwIns]
    prob_switches: _containers.RepeatedCompositeFieldContainer[SwIns]
    tensor_switches: _containers.RepeatedCompositeFieldContainer[SwIns]
    def __init__(self, nodes: _Optional[_Iterable[_Union[ExplGraphNode, _Mapping]]] = ..., prob_switches: _Optional[_Iterable[_Union[SwIns, _Mapping]]] = ..., tensor_switches: _Optional[_Iterable[_Union[SwIns, _Mapping]]] = ..., operators: _Optional[_Iterable[_Union[SwIns, _Mapping]]] = ...) -> None: ...

class Flag(_message.Message):
    __slots__ = ["key", "value"]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class GoalTerm(_message.Message):
    __slots__ = ["args", "name"]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    args: _containers.RepeatedScalarFieldContainer[str]
    name: str
    def __init__(self, name: _Optional[str] = ..., args: _Optional[_Iterable[str]] = ...) -> None: ...

class IndexRange(_message.Message):
    __slots__ = ["index", "range"]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    RANGE_FIELD_NUMBER: _ClassVar[int]
    index: str
    range: int
    def __init__(self, index: _Optional[str] = ..., range: _Optional[int] = ...) -> None: ...

class Option(_message.Message):
    __slots__ = ["flags", "index_range", "tensor_shape"]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    INDEX_RANGE_FIELD_NUMBER: _ClassVar[int]
    TENSOR_SHAPE_FIELD_NUMBER: _ClassVar[int]
    flags: _containers.RepeatedCompositeFieldContainer[Flag]
    index_range: _containers.RepeatedCompositeFieldContainer[IndexRange]
    tensor_shape: _containers.RepeatedCompositeFieldContainer[TensorShape]
    def __init__(self, flags: _Optional[_Iterable[_Union[Flag, _Mapping]]] = ..., index_range: _Optional[_Iterable[_Union[IndexRange, _Mapping]]] = ..., tensor_shape: _Optional[_Iterable[_Union[TensorShape, _Mapping]]] = ...) -> None: ...

class Placeholder(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class PlaceholderData(_message.Message):
    __slots__ = ["goals"]
    GOALS_FIELD_NUMBER: _ClassVar[int]
    goals: _containers.RepeatedCompositeFieldContainer[PlaceholderGoal]
    def __init__(self, goals: _Optional[_Iterable[_Union[PlaceholderGoal, _Mapping]]] = ...) -> None: ...

class PlaceholderGoal(_message.Message):
    __slots__ = ["id", "placeholders", "records"]
    ID_FIELD_NUMBER: _ClassVar[int]
    PLACEHOLDERS_FIELD_NUMBER: _ClassVar[int]
    RECORDS_FIELD_NUMBER: _ClassVar[int]
    id: int
    placeholders: _containers.RepeatedCompositeFieldContainer[Placeholder]
    records: _containers.RepeatedCompositeFieldContainer[DataRecord]
    def __init__(self, id: _Optional[int] = ..., placeholders: _Optional[_Iterable[_Union[Placeholder, _Mapping]]] = ..., records: _Optional[_Iterable[_Union[DataRecord, _Mapping]]] = ...) -> None: ...

class RankRoot(_message.Message):
    __slots__ = ["count", "roots"]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    ROOTS_FIELD_NUMBER: _ClassVar[int]
    count: int
    roots: _containers.RepeatedCompositeFieldContainer[Root]
    def __init__(self, roots: _Optional[_Iterable[_Union[Root, _Mapping]]] = ..., count: _Optional[int] = ...) -> None: ...

class Root(_message.Message):
    __slots__ = ["id", "sorted_id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    SORTED_ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    sorted_id: int
    def __init__(self, id: _Optional[int] = ..., sorted_id: _Optional[int] = ...) -> None: ...

class SwIns(_message.Message):
    __slots__ = ["id", "inside", "name", "sw_type", "values"]
    ID_FIELD_NUMBER: _ClassVar[int]
    INSIDE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SW_TYPE_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    id: int
    inside: float
    name: str
    sw_type: SwType
    values: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, id: _Optional[int] = ..., name: _Optional[str] = ..., values: _Optional[_Iterable[str]] = ..., inside: _Optional[float] = ..., sw_type: _Optional[_Union[SwType, str]] = ...) -> None: ...

class TensorShape(_message.Message):
    __slots__ = ["shape", "tensor_name", "type"]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    TENSOR_NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    tensor_name: str
    type: str
    def __init__(self, tensor_name: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., type: _Optional[str] = ...) -> None: ...

class SwType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
