from sys import modules, _getframe
from os import scandir
from os.path import join, dirname, splitext, isfile, isdir
from pathlib import PurePath
from importlib import import_module, util
from types import ModuleType, SimpleNamespace
from itertools import chain
from pkgutil import walk_packages
import pprint
from inspect import stack
import inspect

def stack(max_depth: int = None):
    """Fast alternative to `inspect.stack()`

    Use optional `max_depth` to limit search depth
    Based on: github.com/python/cpython/blob/3.11/Lib/inspect.py

    Compared to `inspect.stack()`:
     * Does not read source files to load neighboring context
     * Less accurate filename determination, still correct for most cases
     * Does not compute 3.11+ code positions (PEP 657)

    Compare:

    In [3]: %timeit stack_depth(100, lambda: inspect.stack())
    67.7 ms ± 1.35 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

    In [4]: %timeit stack_depth(100, lambda: inspect.stack(0))
    22.7 ms ± 747 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

    In [5]: %timeit stack_depth(100, lambda: fast_stack())
    108 µs ± 180 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

    In [6]: %timeit stack_depth(100, lambda: fast_stack(10))
    14.1 µs ± 33.4 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    """
    def frame_infos(frame: FrameType | None):
        while frame := frame and frame.f_back:
            yield inspect.FrameInfo(
                frame,
                inspect.getfile(frame),
                frame.f_lineno,
                frame.f_code.co_name,
                None, None,
            )

    try :
      stack = list(it.islice(frame_infos(inspect.currentframe()), max_depth))
      pprint.pp("Success")
    except :
      pprint.pp("Failure")
      stack = inspect.stack()
    return stack

def get_module_from_frame(frame) :
  try :
    module = modules[frame.f_globals["__name__"]]
    pprint("get_module_from_frame succeeded")
    return module
  except :
    pprint("get_module_from_frame failed")
    return inspect.getmodule(frame)

def caller_name(depth = 2):
  """Get a module of a caller
     `depth` specifies how many levels of stack to skip while getting caller
     name. depth=1 means "who calls me", depth=2 "who calls my caller" etc.

     https://gist.github.com/techtonik/2151727
  """
  stack = stack(2)
  start = 0 + skip
  if len(stack) < start + 1:
    raise Exception("Stack limit reached")
  previous_frame = stack[start][0]
  return get_module_from_frame(previous_frame)

def caller_info():
  previous_frame = None
  previous_frame = inspect.currentframe().f_back.f_back
  module_name = previous_frame.f_globals["__name__"]
  return modules[module_name]
  try :
    previous_frame = _getframe(2)
  except Exception :
    previous_frame = inspect.currentframe().f_back.f_back
  finally :
    try :
      module = modules[module_name]
    finally :
      # https://bugs.python.org/issue543148
      del previous_frame
  return module

def lazy_load_module(module_name) :
  if module_name in sys.modules:
    print(f"{module_name} already in sys.modules")
    return sys.modules[module_name]
  if (spec := util.find_spec(module_name)) is not None :
    module = util.module_from_spec(spec)
    loader = util.LazyLoader(spec.loader)
    spec.loader = loader
    loader.exec_module(module)
    modules[name] = module
    return module_name
  print("Can't lazy load module")

def load_module(module_name) :
  if module_name in sys.modules:
    print(f"{module_name} already in sys.modules")
    return sys.modules[module_name]
  if (spec := util.find_spec(module_name)) is not None :
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    modules[name] = module
    return module_name
  print("Can't load module")

def snake_to_camel(word) :
  return ''.join(x.capitalize() or '_' for x in word.split('_'))

def camel_to_snake(s) :
  return ''.join(['_'+c.lower() if c.isupper() else c for c in s]).lstrip('_')

def all_files_in_path(*args, **kwargs) :
  if not isinstance(args[0], str) :
    raise RuntimeError("all_files_in_path failed because package name is missing") from e
  package_path = args[0]
  package_file = "__init__.py"
  exclude_files = kwargs.setdefault("exclude_files", [package_file])
  extension = kwargs.setdefault("extension", None)
  path_from_package = kwargs.setdefault("path_from_package", "")
  path = package_path if not path_from_package else join(package_path, path_from_package)
  if extension is not None :
    extension = extension.lower()
  path_from_package_dot_notation = '.'.join(PurePath(path_from_package).parts)
  dict = {}
  entries = scandir(path)
  for entry in entries :
    entry_name = entry.name
    if isdir(entry) :
      kwargs["path_from_package"] = join(path_from_package, entry_name)
      if isfile(join(package_path, kwargs["path_from_package"], package_file)) :
        dict['.'.join(filter(None, [path_from_package_dot_notation, entry_name]))] = [entry_name]
      else :
        dict.update(all_files_in_path(package_path, **kwargs))
    elif entry_name not in exclude_files :
      file_name, file_extension = splitext(entry_name)
      if extension is None or file_extension.lower() == extension :
        dict['.'.join(filter(None, [path_from_package_dot_notation, file_name]))] = [file_name]
  return dict

class LazyModule(ModuleType) :
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, *args, **kwargs) :
      package_name, package_file, *_ = list(args) + [None] * 2
      if not isinstance(package_name, str) :
        raise RuntimeError("Autoload failed because package name is missing or not a string") from e
      if not isinstance(package_file, str) :
        raise RuntimeError("Autoload failed because package file name is missing or not a string") from e
      package_spec = kwargs.get("package_spec", None)
      import_structure = kwargs.get("import_structure", None)
      extra_objects = kwargs.get("extra_objects", None)
      super().__init__(package_name)
      package_dir = dirname(package_file)
      if import_structure is None :
        import_structure = all_files_in_path(package_dir, extension = ".py")
      modules = import_structure.keys()
      classes = import_structure.values()
      self._modules = set(modules)
      self.__class_to_module = {}
      for module, classlist in import_structure.items():
        for class_name in classlist:
          self.__class_to_module[class_name] = module
      # Needed for autocompletion in an IDE
      self.__all__ = list(modules) + list(chain(*classes))
      self.__file__ = package_file
      self.__spec__ = package_spec
      self.__path__ = [package_dir]
      self.__objects = {} if extra_objects is None else extra_objects
      self.__package__ = package_name
      self.__import_structure = import_structure
      name_with_dot = self.__name__+'.'
      #for loader, module_name, is_pkg in walk_packages(self.__path__, name_with_dot):
      #  sub_package_name = module_name.replace(name_with_dot, '')
      #  sub_package = self.__get_module(sub_package_name)
      #  setattr(self, sub_package_name, sub_package)
      #  self.__all__.append(sub_package)
      #  print(sub_package_name)

    # Needed for autocompletion in an IDE
    def __dir__(self) :
      result = super().__dir__()
      # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
      # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
      for attr in self.__all__:
        if attr not in result :
          result.append(attr)
      return result

    def __getattr__(self, module_name: str) :
      if module_name in self.__objects :
        return self.__objects[module_name]
      if (spec := util.find_spec(module_name)) is not None :
        value = lazy_load_module(module_name)
        setattr(self, module_name, value)
        return value
      if module_name in self.__class_to_module.keys() :
        module = self.__get_module(self.__class_to_module[module_name])
        value = module if module_name.lower() == module_name else getattr(module, module_name)
      elif module_name in self._modules :
        value = self.__get_module(module_name)
      else :
        raise AttributeError(f"Package {self.__name__} has no module {name}")
      modules[self.__name__ + '.' + module_name] = value
      setattr(self, module_name, value)
      return value


    def __get_module(self, name: str) :
      try :
        name = "." + name
        module = import_module(name, self.__name__)
        return module
      except Exception as e :
        raise RuntimeError(
          f"Failed to import {self.__name__}.{name} because of the following error (look up to see its"
          f" traceback):\n{e}"
        ) from e

    def __reduce__(self) :
      return (self.__class__, (
        self.__package__,
        self.__file__,
        self.__import_structure
      ))


def AutoLoad(**kwargs) :
  module = caller_name()
  pprint.pp(module)
  module_name = module.__name__
  module_file = module.__file__
  module_spec = util.find_spec(module_name)
  module = LazyModule(module_name, module_file, spec = module_spec, **kwargs)
  modules[module_name] = module
  return module
