from sys import modules
from os import getcwd
from os.path import join
from .util import (
#    LazyModule,
    import_structure
)

import_test = import_structure(join(getcwd(), "class"))

#modules[__name__] = LazyModule(
#  __name__,
#  globals()["__file__"],
#  import_structure(join(getcwd(), "class")),
#  module_spec = __spec__
#)
