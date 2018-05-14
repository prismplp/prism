# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: expl.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='expl.proto',
  package='prism',
  syntax='proto3',
  serialized_pb=_b('\n\nexpl.proto\x12\x05prism\"T\n\tExplGraph\x12#\n\x05goals\x18\x01 \x03(\x0b\x32\x14.prism.ExplGraphGoal\x12\"\n\troot_list\x18\x02 \x03(\x0b\x32\x0f.prism.RankRoot\"X\n\rExplGraphGoal\x12\"\n\x04node\x18\x01 \x01(\x0b\x32\x14.prism.ExplGraphNode\x12#\n\x05paths\x18\x02 \x03(\x0b\x32\x14.prism.ExplGraphPath\"O\n\rExplGraphPath\x12#\n\x05nodes\x18\x01 \x03(\x0b\x32\x14.prism.ExplGraphNode\x12\x19\n\x03sws\x18\x02 \x03(\x0b\x32\x0c.prism.SwIns\"M\n\rExplGraphNode\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x11\n\tsorted_id\x18\x02 \x01(\x05\x12\x1d\n\x04goal\x18\x03 \x01(\x0b\x32\x0f.prism.GoalTerm\"&\n\x08GoalTerm\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04\x61rgs\x18\x02 \x03(\t\"\x15\n\x05Value\x12\x0c\n\x04list\x18\x01 \x03(\t\">\n\x05SwIns\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x1b\n\x05value\x18\x03 \x01(\x0b\x32\x0c.prism.Value\"%\n\x04Root\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x11\n\tsorted_id\x18\x02 \x01(\x05\"5\n\x08RankRoot\x12\x1a\n\x05roots\x18\x01 \x03(\x0b\x32\x0b.prism.Root\x12\r\n\x05\x63ount\x18\x02 \x01(\x05\x62\x06proto3')
)




_EXPLGRAPH = _descriptor.Descriptor(
  name='ExplGraph',
  full_name='prism.ExplGraph',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='goals', full_name='prism.ExplGraph.goals', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='root_list', full_name='prism.ExplGraph.root_list', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21,
  serialized_end=105,
)


_EXPLGRAPHGOAL = _descriptor.Descriptor(
  name='ExplGraphGoal',
  full_name='prism.ExplGraphGoal',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='node', full_name='prism.ExplGraphGoal.node', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='paths', full_name='prism.ExplGraphGoal.paths', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=107,
  serialized_end=195,
)


_EXPLGRAPHPATH = _descriptor.Descriptor(
  name='ExplGraphPath',
  full_name='prism.ExplGraphPath',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='nodes', full_name='prism.ExplGraphPath.nodes', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sws', full_name='prism.ExplGraphPath.sws', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=197,
  serialized_end=276,
)


_EXPLGRAPHNODE = _descriptor.Descriptor(
  name='ExplGraphNode',
  full_name='prism.ExplGraphNode',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='prism.ExplGraphNode.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sorted_id', full_name='prism.ExplGraphNode.sorted_id', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='goal', full_name='prism.ExplGraphNode.goal', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=278,
  serialized_end=355,
)


_GOALTERM = _descriptor.Descriptor(
  name='GoalTerm',
  full_name='prism.GoalTerm',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='prism.GoalTerm.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='args', full_name='prism.GoalTerm.args', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=357,
  serialized_end=395,
)


_VALUE = _descriptor.Descriptor(
  name='Value',
  full_name='prism.Value',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='list', full_name='prism.Value.list', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=397,
  serialized_end=418,
)


_SWINS = _descriptor.Descriptor(
  name='SwIns',
  full_name='prism.SwIns',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='prism.SwIns.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='prism.SwIns.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='prism.SwIns.value', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=420,
  serialized_end=482,
)


_ROOT = _descriptor.Descriptor(
  name='Root',
  full_name='prism.Root',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='prism.Root.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sorted_id', full_name='prism.Root.sorted_id', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=484,
  serialized_end=521,
)


_RANKROOT = _descriptor.Descriptor(
  name='RankRoot',
  full_name='prism.RankRoot',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='roots', full_name='prism.RankRoot.roots', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='count', full_name='prism.RankRoot.count', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=523,
  serialized_end=576,
)

_EXPLGRAPH.fields_by_name['goals'].message_type = _EXPLGRAPHGOAL
_EXPLGRAPH.fields_by_name['root_list'].message_type = _RANKROOT
_EXPLGRAPHGOAL.fields_by_name['node'].message_type = _EXPLGRAPHNODE
_EXPLGRAPHGOAL.fields_by_name['paths'].message_type = _EXPLGRAPHPATH
_EXPLGRAPHPATH.fields_by_name['nodes'].message_type = _EXPLGRAPHNODE
_EXPLGRAPHPATH.fields_by_name['sws'].message_type = _SWINS
_EXPLGRAPHNODE.fields_by_name['goal'].message_type = _GOALTERM
_SWINS.fields_by_name['value'].message_type = _VALUE
_RANKROOT.fields_by_name['roots'].message_type = _ROOT
DESCRIPTOR.message_types_by_name['ExplGraph'] = _EXPLGRAPH
DESCRIPTOR.message_types_by_name['ExplGraphGoal'] = _EXPLGRAPHGOAL
DESCRIPTOR.message_types_by_name['ExplGraphPath'] = _EXPLGRAPHPATH
DESCRIPTOR.message_types_by_name['ExplGraphNode'] = _EXPLGRAPHNODE
DESCRIPTOR.message_types_by_name['GoalTerm'] = _GOALTERM
DESCRIPTOR.message_types_by_name['Value'] = _VALUE
DESCRIPTOR.message_types_by_name['SwIns'] = _SWINS
DESCRIPTOR.message_types_by_name['Root'] = _ROOT
DESCRIPTOR.message_types_by_name['RankRoot'] = _RANKROOT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ExplGraph = _reflection.GeneratedProtocolMessageType('ExplGraph', (_message.Message,), dict(
  DESCRIPTOR = _EXPLGRAPH,
  __module__ = 'expl_pb2'
  # @@protoc_insertion_point(class_scope:prism.ExplGraph)
  ))
_sym_db.RegisterMessage(ExplGraph)

ExplGraphGoal = _reflection.GeneratedProtocolMessageType('ExplGraphGoal', (_message.Message,), dict(
  DESCRIPTOR = _EXPLGRAPHGOAL,
  __module__ = 'expl_pb2'
  # @@protoc_insertion_point(class_scope:prism.ExplGraphGoal)
  ))
_sym_db.RegisterMessage(ExplGraphGoal)

ExplGraphPath = _reflection.GeneratedProtocolMessageType('ExplGraphPath', (_message.Message,), dict(
  DESCRIPTOR = _EXPLGRAPHPATH,
  __module__ = 'expl_pb2'
  # @@protoc_insertion_point(class_scope:prism.ExplGraphPath)
  ))
_sym_db.RegisterMessage(ExplGraphPath)

ExplGraphNode = _reflection.GeneratedProtocolMessageType('ExplGraphNode', (_message.Message,), dict(
  DESCRIPTOR = _EXPLGRAPHNODE,
  __module__ = 'expl_pb2'
  # @@protoc_insertion_point(class_scope:prism.ExplGraphNode)
  ))
_sym_db.RegisterMessage(ExplGraphNode)

GoalTerm = _reflection.GeneratedProtocolMessageType('GoalTerm', (_message.Message,), dict(
  DESCRIPTOR = _GOALTERM,
  __module__ = 'expl_pb2'
  # @@protoc_insertion_point(class_scope:prism.GoalTerm)
  ))
_sym_db.RegisterMessage(GoalTerm)

Value = _reflection.GeneratedProtocolMessageType('Value', (_message.Message,), dict(
  DESCRIPTOR = _VALUE,
  __module__ = 'expl_pb2'
  # @@protoc_insertion_point(class_scope:prism.Value)
  ))
_sym_db.RegisterMessage(Value)

SwIns = _reflection.GeneratedProtocolMessageType('SwIns', (_message.Message,), dict(
  DESCRIPTOR = _SWINS,
  __module__ = 'expl_pb2'
  # @@protoc_insertion_point(class_scope:prism.SwIns)
  ))
_sym_db.RegisterMessage(SwIns)

Root = _reflection.GeneratedProtocolMessageType('Root', (_message.Message,), dict(
  DESCRIPTOR = _ROOT,
  __module__ = 'expl_pb2'
  # @@protoc_insertion_point(class_scope:prism.Root)
  ))
_sym_db.RegisterMessage(Root)

RankRoot = _reflection.GeneratedProtocolMessageType('RankRoot', (_message.Message,), dict(
  DESCRIPTOR = _RANKROOT,
  __module__ = 'expl_pb2'
  # @@protoc_insertion_point(class_scope:prism.RankRoot)
  ))
_sym_db.RegisterMessage(RankRoot)


# @@protoc_insertion_point(module_scope)