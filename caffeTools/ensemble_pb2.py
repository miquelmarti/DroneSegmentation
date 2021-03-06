# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ensemble.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='ensemble.proto',
  package='ensemble',
  serialized_pb=_b('\n\x0e\x65nsemble.proto\x12\x08\x65nsemble\"j\n\x05Model\x12\x0e\n\x06\x64\x65ploy\x18\x01 \x02(\t\x12\x0f\n\x07weights\x18\x02 \x02(\t\x12\x13\n\x05input\x18\x03 \x01(\t:\x04\x64\x61ta\x12\x15\n\x06output\x18\x04 \x01(\t:\x05score\x12\x14\n\tweighting\x18\x05 \x01(\x05:\x01\x31\"3\n\x0bLogitFolder\x12\x0e\n\x06\x66older\x18\x01 \x02(\t\x12\x14\n\tweighting\x18\x02 \x01(\x05:\x01\x31\"\x8c\x02\n\x05Input\x12\x0c\n\x04\x66ile\x18\x01 \x01(\t\x12/\n\x04type\x18\x02 \x02(\x0e\x32\x19.ensemble.Input.InputType:\x06LABELS\x12\x0f\n\x07\x63olours\x18\x03 \x02(\t\x12(\n\x04mean\x18\x04 \x01(\x0b\x32\x1a.ensemble.Input.MeanValues\x12\x15\n\x06resize\x18\x05 \x01(\x08:\x05\x66\x61lse\x1a\x36\n\nMeanValues\x12\x0c\n\x01r\x18\x01 \x02(\x02:\x01\x30\x12\x0c\n\x01g\x18\x02 \x02(\x02:\x01\x30\x12\x0c\n\x01\x62\x18\x03 \x02(\x02:\x01\x30\":\n\tInputType\x12\t\n\x05VIDEO\x10\x01\x12\n\n\x06IMAGES\x10\x02\x12\n\n\x06LABELS\x10\x03\x12\n\n\x06WEBCAM\x10\x04\"\xc3\x01\n\x08\x45nsemble\x12\x35\n\rensemble_type\x18\x01 \x01(\x0e\x32\x16.ensemble.EnsembleType:\x06VOTING\x12\x1e\n\x05model\x18\x02 \x03(\x0b\x32\x0f.ensemble.Model\x12*\n\x0blogitFolder\x18\x03 \x03(\x0b\x32\x15.ensemble.LogitFolder\x12\x1e\n\x05input\x18\x04 \x02(\x0b\x32\x0f.ensemble.Input\x12\x14\n\x0coutputFolder\x18\x05 \x01(\t*R\n\x0c\x45nsembleType\x12\n\n\x06VOTING\x10\x01\x12\x0c\n\x08LOGITARI\x10\x02\x12\x0c\n\x08LOGITGEO\x10\x03\x12\x0c\n\x08PROBAARI\x10\x04\x12\x0c\n\x08PROBAGEO\x10\x05')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

_ENSEMBLETYPE = _descriptor.EnumDescriptor(
  name='EnsembleType',
  full_name='ensemble.EnsembleType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='VOTING', index=0, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOGITARI', index=1, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOGITGEO', index=2, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PROBAARI', index=3, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PROBAGEO', index=4, number=5,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=658,
  serialized_end=740,
)
_sym_db.RegisterEnumDescriptor(_ENSEMBLETYPE)

EnsembleType = enum_type_wrapper.EnumTypeWrapper(_ENSEMBLETYPE)
VOTING = 1
LOGITARI = 2
LOGITGEO = 3
PROBAARI = 4
PROBAGEO = 5


_INPUT_INPUTTYPE = _descriptor.EnumDescriptor(
  name='InputType',
  full_name='ensemble.Input.InputType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='VIDEO', index=0, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='IMAGES', index=1, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LABELS', index=2, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='WEBCAM', index=3, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=400,
  serialized_end=458,
)
_sym_db.RegisterEnumDescriptor(_INPUT_INPUTTYPE)


_MODEL = _descriptor.Descriptor(
  name='Model',
  full_name='ensemble.Model',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='deploy', full_name='ensemble.Model.deploy', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='weights', full_name='ensemble.Model.weights', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='input', full_name='ensemble.Model.input', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("data").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='output', full_name='ensemble.Model.output', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("score").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='weighting', full_name='ensemble.Model.weighting', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28,
  serialized_end=134,
)


_LOGITFOLDER = _descriptor.Descriptor(
  name='LogitFolder',
  full_name='ensemble.LogitFolder',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='folder', full_name='ensemble.LogitFolder.folder', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='weighting', full_name='ensemble.LogitFolder.weighting', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=136,
  serialized_end=187,
)


_INPUT_MEANVALUES = _descriptor.Descriptor(
  name='MeanValues',
  full_name='ensemble.Input.MeanValues',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='r', full_name='ensemble.Input.MeanValues.r', index=0,
      number=1, type=2, cpp_type=6, label=2,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='g', full_name='ensemble.Input.MeanValues.g', index=1,
      number=2, type=2, cpp_type=6, label=2,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='b', full_name='ensemble.Input.MeanValues.b', index=2,
      number=3, type=2, cpp_type=6, label=2,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=344,
  serialized_end=398,
)

_INPUT = _descriptor.Descriptor(
  name='Input',
  full_name='ensemble.Input',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='file', full_name='ensemble.Input.file', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='type', full_name='ensemble.Input.type', index=1,
      number=2, type=14, cpp_type=8, label=2,
      has_default_value=True, default_value=3,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='colours', full_name='ensemble.Input.colours', index=2,
      number=3, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mean', full_name='ensemble.Input.mean', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='resize', full_name='ensemble.Input.resize', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_INPUT_MEANVALUES, ],
  enum_types=[
    _INPUT_INPUTTYPE,
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=190,
  serialized_end=458,
)


_ENSEMBLE = _descriptor.Descriptor(
  name='Ensemble',
  full_name='ensemble.Ensemble',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ensemble_type', full_name='ensemble.Ensemble.ensemble_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='model', full_name='ensemble.Ensemble.model', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='logitFolder', full_name='ensemble.Ensemble.logitFolder', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='input', full_name='ensemble.Ensemble.input', index=3,
      number=4, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='outputFolder', full_name='ensemble.Ensemble.outputFolder', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=461,
  serialized_end=656,
)

_INPUT_MEANVALUES.containing_type = _INPUT
_INPUT.fields_by_name['type'].enum_type = _INPUT_INPUTTYPE
_INPUT.fields_by_name['mean'].message_type = _INPUT_MEANVALUES
_INPUT_INPUTTYPE.containing_type = _INPUT
_ENSEMBLE.fields_by_name['ensemble_type'].enum_type = _ENSEMBLETYPE
_ENSEMBLE.fields_by_name['model'].message_type = _MODEL
_ENSEMBLE.fields_by_name['logitFolder'].message_type = _LOGITFOLDER
_ENSEMBLE.fields_by_name['input'].message_type = _INPUT
DESCRIPTOR.message_types_by_name['Model'] = _MODEL
DESCRIPTOR.message_types_by_name['LogitFolder'] = _LOGITFOLDER
DESCRIPTOR.message_types_by_name['Input'] = _INPUT
DESCRIPTOR.message_types_by_name['Ensemble'] = _ENSEMBLE
DESCRIPTOR.enum_types_by_name['EnsembleType'] = _ENSEMBLETYPE

Model = _reflection.GeneratedProtocolMessageType('Model', (_message.Message,), dict(
  DESCRIPTOR = _MODEL,
  __module__ = 'ensemble_pb2'
  # @@protoc_insertion_point(class_scope:ensemble.Model)
  ))
_sym_db.RegisterMessage(Model)

LogitFolder = _reflection.GeneratedProtocolMessageType('LogitFolder', (_message.Message,), dict(
  DESCRIPTOR = _LOGITFOLDER,
  __module__ = 'ensemble_pb2'
  # @@protoc_insertion_point(class_scope:ensemble.LogitFolder)
  ))
_sym_db.RegisterMessage(LogitFolder)

Input = _reflection.GeneratedProtocolMessageType('Input', (_message.Message,), dict(

  MeanValues = _reflection.GeneratedProtocolMessageType('MeanValues', (_message.Message,), dict(
    DESCRIPTOR = _INPUT_MEANVALUES,
    __module__ = 'ensemble_pb2'
    # @@protoc_insertion_point(class_scope:ensemble.Input.MeanValues)
    ))
  ,
  DESCRIPTOR = _INPUT,
  __module__ = 'ensemble_pb2'
  # @@protoc_insertion_point(class_scope:ensemble.Input)
  ))
_sym_db.RegisterMessage(Input)
_sym_db.RegisterMessage(Input.MeanValues)

Ensemble = _reflection.GeneratedProtocolMessageType('Ensemble', (_message.Message,), dict(
  DESCRIPTOR = _ENSEMBLE,
  __module__ = 'ensemble_pb2'
  # @@protoc_insertion_point(class_scope:ensemble.Ensemble)
  ))
_sym_db.RegisterMessage(Ensemble)


# @@protoc_insertion_point(module_scope)
