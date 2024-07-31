from sys import modules
from os import getcwd
from .util import (
    LazyModule,
    import_structure
)

modules[__name__] = LazyModule(
  __name__,
  globals()["__file__"], import_structure(getcwd()),
  module_spec = __spec__
)
