import sys
from os import scandir
from os.path import join, dirname, splitext, isfile, isdir
from pathlib import PurePath
from importlib import import_module, util
from types import ModuleType, FrameType
from itertools import chain, islice
import pprint
from inspect import stack
import inspect
from typing import Any

def unpack(*args, default : Any = None, items : int = 1) -> list[Any] :
  return list(args) + [None] * items

def get_stack(max_depth : int = None) :
  """
  Fast alternative to `inspect.stack()`
  Use optional `max_depth` to limit search depth

  Based on :
  https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
  https://github.com/python/cpython/blob/3.11/Lib/inspect.py

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
  def frame_info(frame: FrameType | None):
    while frame := frame and frame.f_back:
      yield inspect.FrameInfo(
        frame,
        inspect.getfile(frame),
        frame.f_lineno,
        frame.f_code.co_name,
        None, None,
      )
  try :
    stack = list(islice(frame_info(get_frame()), max_depth))
  except Exception as e :
    # Fallback to `inspect.stack()` in case of error
    stack = inspect.stack()
    # Remove 1 frame from the start because of extra call to this wrapper
    # Remove frames at the end to keep into account `max_depth` calue
    stack = stack[1:(max_depth+1)]
  finally :
    return stack

def get_frame(depth: int = 0) :
  """
  Get a frame at a certain depth
  """
  try :
    # Fairly fast, but internal function
    # Add 1 to the depth to compensate for this wrapper function
    return sys._getframe(depth + 1)
  except Exception as e :
    # Fallback in case `sys._getframe` is not available
    # Use `f_back` to get earlier frames as far as needed
    frame = inspect.currentframe().f_back
    while depth > 0:
      frame = frame.f_back
      depth = depth - 1
    return frame

def get_module_from_frame(frame) :
  """
  Retrieve a module from a `frame`
  """
  try :
    return sys.modules[frame.f_globals["__name__"]]
  except Exception as e :
    # Fallback in case f_globals not available
    return inspect.getmodule(frame)

def get_caller_module(depth : int = 1) -> ModuleType :
  """
  Get a module of a caller
  `depth` specifies how many levels of stack to skip while getting caller
  name. depth=1 means "who calls me", depth=2 "who calls my caller" etc.

  Based on https://gist.github.com/techtonik/2151727
  """
  depth = depth + 1
  try :
    previous_frame = get_frame(depth)
  except Exception as e :
    stack_size = depth + 1
    stack = get_stack(stack_size)
    if len(stack) < stack_size:
      raise Exception("Stack limit reached")
    previous_frame = stack[depth][0]
  finally :
    # https://bugs.python.org/issue543148
    module = get_module_from_frame(previous_frame)
    del previous_frame
    return module

def get_module_from_code(code):
  print(code)
  def run_code(fullname, source_code = None):
    spec = util.spec_from_loader(fullname, loader = None)
    module = util.module_from_spec(spec)
    exec(source_code if source_code else fullname, module.__dict__)
    return module
  try:
    return sys.modules[code]
  except KeyError:
    mod = run_code(code)
    sys.modules[code] = mod
    return mod

def snake_to_camel(word : str) -> str :
  return ''.join(x.capitalize() or '_' for x in word.split('_'))

def camel_to_snake(s : str) -> str :
  return ''.join(['_'+c.lower() if c.isupper() else c for c in s]).lstrip('_')

def all_files_in_path(*args, **kwargs) :
  if not args[0] :
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
        dict['.' + '.'.join(filter(None, [path_from_package_dot_notation, entry_name]))] = [entry_name]
      else :
        dict.update(all_files_in_path(package_path, **kwargs))
    elif entry_name not in exclude_files :
      file_name, file_extension = splitext(entry_name)
      if extension is None or file_extension.lower() == extension :
        dict['.' + '.'.join(filter(None, [path_from_package_dot_notation, file_name]))] = [file_name]
  return dict

class LazyModule(ModuleType) :
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.

    Based on :
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/import_utils.py
    https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    """
    def __init__(self, *args, **kwargs) :
      module, *_ = unpack(*args)
      import_structure = kwargs.get("import_structure", None)
      extra_objects = kwargs.get("extra_objects", None)
      module_dir = dirname(module.__file__)
      super().__init__(module.__name__)
      if import_structure is None :
        import_structure = all_files_in_path(module_dir, extension = ".py")
      self.__LAZY_MODULE__class_to_module = {}
      if import_structure :
        modules = import_structure.keys()
        classes = import_structure.values()
        for module_name, classlist in import_structure.items():
          for class_name in classlist:
            self.__LAZY_MODULE__class_to_module[class_name] = module_name
      else :
        modules = []
        classes = []
      self.__LAZY_MODULE__modules = set(modules)
      # Needed for autocompletion in an IDE
      self.__all__ = list(modules) + list(chain(*classes))
      self.__spec__ = module.__spec__
      self.__file__ = module.__file__
      self.__loader__ = module.__loader__
      self.__path__ = [module_dir]
      self.__package__ = module.__name__.split('.')[0]
      self.__LAZY_MODULE__import_structure = import_structure
      self.__LAZY_MODULE__objects = {} if extra_objects is None else extra_objects

    # Needed for autocompletion in an IDE
    def __dir__(self) :
      result = super().__dir__()
      # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
      # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
      for attr in self.__all__:
        if attr not in result :
          result.append(attr)
      return result

    def load(self, name: str) :
      if hasattr(self, '__LAZY_MODULE__module__'+name) :
        return getattr(self, '__LAZY_MODULE__module__'+name)
      if name in self.__LAZY_MODULE__class_to_module.keys() :
        module_name = self.__LAZY_MODULE__class_to_module[name]
        if module_name[0] != '.' :
          source_code = f"from {module_name} import {name}"
          module = import_from_string(module_name+'.'+name, source_code)
          value = getattr(module, name)
          sys.modules[self.__name__ + '.' + name] = module
          setattr(self, '__LAZY_MODULE__module__'+name, value)
          return value
      return getattr(self, '__LAZY_MODULE__module__'+name)

    def __getattr__(self, name: str) :
      if name in self.__LAZY_MODULE__objects :
        return self.__LAZY_MODULE__objects[name]
      full_name = name if name[0] != '.' else self.__name__ + name
      if full_name in sys.modules :
        return sys.modules[full_name]
      if name in self.__LAZY_MODULE__class_to_module.keys() :
        module_name = self.__LAZY_MODULE__class_to_module[name]
        module = self.__get_module(self.__name__+module_name)
        value = module if name.lower() == name else getattr(module, name)
      elif full_name in self.__LAZY_MODULE__modules :
        value = self.__get_module(full_name)
      elif f".{name}" in self.__LAZY_MODULE__modules :
        value = self.__get_module(self.__name__+name)
      else :
        raise AttributeError(f"Attribute {name} unknown for module {self.__name__}.")
      sys.modules[full_name] = value
      setattr(self, name, value)
      return value


    def __get_module(self, name: str) :
      #try :
        module = import_module(name)
        spec = module.__spec__
        spec.loader.exec_module(module)
        return module
      #except Exception as e :
      #  raise RuntimeError(
      #    f"Failed to import {self.__name__}.{name} because of the following error (look up to see its"
      #    f" traceback):\n{e}"
      #  ) from e

    def __reduce__(self) :
      return (self.__class__, (
        self.__name__,
        self.__file__,
        self.__LAZY_MODULE__import_structure
      ))


def AutoLoad(**kwargs) :
  module = get_caller_module()
  module_name = module.__name__
  #module_name = '.'.join(filter(None, [module.__package__, module.__name__]))
  module = LazyModule(module, **kwargs)
  sys.modules[module_name] = module
  return module



























class Module_Attr:
  def __init__(self, value):
    self.name = value
    print(f"INIT --  self.name = {value}")
  def __call__(self, instance, *args, **kwargs):
    print(instance._Module_Attr__PROXY__activated)
    print(self.name)
    if not instance._Module_Attr__PROXY__activated :
      instance._Module_Attr__PROXY__activate()
    return getattr(instance.__module, self.name)(*args, **kwargs)
    print(f"CALL --  instance.__dict__[{self.name}]([{args}], {kwargs})")
  def __get__(self, instance, owner):
    print(instance._Module_Attr__PROXY__activated)
    print(f"GET --  instance.__dict__[{self.name}]")
    if not instance._Module_Attr__PROXY__activated :
      return instance.__module_proxy[self.name]
    else :
      return getattr(instance.__module, self.name)


def get_mod(fullname, attrs = None):
  if not attrs :
    code = f"from {fullname} import *"
    return get_module_from_code(code)
  if isinstance(attrs, str) :
    code = f"from {fullname} import {attrs}"
    return getattr(get_module_from_code(code), attrs)
  code = f"from {fullname} import {', '.join(attrs)}"
  return (getattr(get_module_from_code(code), attr) for attr in attrs)

















def module(name, attrs = None) :

  class Module_proxy(object):
    attr_names = []
    _Module_Attr__PROXY__activated = False
    _Module_Attr__module_proxy = {}
    _Module_Attr__module = []
    MODULY_PROXY_name = ''
    parent = None

    @classmethod
    def _Module_Attr__PROXY__activate(cls) :
      if not Module_proxy._Module_Attr__PROXY__activated :
        Module_proxy._Module_Attr__PROXY__activated = True
        print("ACTIVATE")
        mod = get_mod(cls.parent.MODULY_PROXY_name, cls.attr_names)
        Module_proxy._Module_Attr__module = mod
        print(Module_proxy._Module_Attr__module)
        if Module_proxy.attr_names :
          Module_proxy._Module_Attr__module = []
          for key in Module_proxy.attr_names :
            attrval = next(mod)
            """
            if callable(attrval) :
              def q(cls, *args, **kwargs) :
                return attrval(*args, **kwargs)
            else :
            """
            q = attrval
            Module_proxy._Module_Attr__module.append(q)
            setattr(parent, key, q)

    def __init__(self, name) :
      self.MODULY_PROXY_name = name

  class Module_proxy_child(Module_proxy):
    @classmethod

    def setup(cls, name, attrs = None):
      proxy = cls(name)

    def __getattr__(self, key):
      self._Module_Attr__PROXY__activate()
      return getattr(getattr(Module_proxy._Module_Attr__module, self.MODULY_PROXY_name), key)

    def __str__(self):
      self._Module_Attr__PROXY__activate()
      return str(getattr(Module_proxy._Module_Attr__module, self.MODULY_PROXY_name))

    def __call__(self, *args, **kwargs):
      self._Module_Attr__PROXY__activate()
      print(Module_proxy.attr_names)
      print(Module_proxy._Module_Attr__module)
      print(Module_proxy._Module_Attr__module_proxy)
      return getattr(Module_proxy._Module_Attr__module, self.MODULY_PROXY_name)(*args, **kwargs)


  class Module_proxy_parent(Module_proxy):

    @classmethod
    def setup(cls, name, attrs = None):
      proxy = cls(name)
      Module_proxy.parent = proxy
      if not attrs :
        return proxy
      if isinstance(attrs, str) :
        attrs = [attrs]
      for attr in attrs :
        a = Module_Attr(attr)
        setattr(Module_proxy_parent, attr, a)
        child = Module_proxy_child(attr)
        Module_proxy.attr_names.append(attr)
        Module_proxy._Module_Attr__module.append(a)
        Module_proxy._Module_Attr__module[-1] = child
        Module_proxy._Module_Attr__module_proxy[attr] = child
      return proxy

    def __getattr__(self, key):
      self._Module_Attr__PROXY__activate()
      try :
        return getattr(Module_proxy._Module_Attr__module, key)
      except Exception as e :
        print(Module_proxy.attr_names)
        print(Module_proxy._Module_Attr__module)
        return getattr(Module_proxy._Module_Attr__module, Module_proxy.attr_names[0])[key]

    def __getitem__(self, key):
      return type(self)._Module_Attr__module[key]

    def __call__(self, *args, **kwargs):
      self._Module_Attr__PROXY__activate()
      return getattr(Module_proxy._Module_Attr__module, Module_proxy.attr_names[0])(*args, **kwargs)


  proxy = Module_proxy_parent.setup(name, attrs)
  return proxy
