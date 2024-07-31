from os import getcwd
from .util import (
    LazyModule,
    import_structure
)

sys.modules[__name__] = LazyModule(
  __name__,
  globals()["__file__"], import_structure(getcwd()),
  module_spec = __spec__
)
