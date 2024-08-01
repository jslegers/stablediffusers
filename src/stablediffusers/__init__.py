from sys import modules
from os.path import join, dirname
from .util import (
    LazyModule,
    import_structure
)

modules[__name__] = LazyModule(
  __name__,
  globals()["__file__"],
  import_structure(join(dirname(globals()["__file__"]), "class")),
  module_spec = __spec__
)
