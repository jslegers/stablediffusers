from sys import modules, _getframe
from os import scandir
from os.path import join, dirname, splitext, isfile, isdir
from pathlib import PurePath
from importlib import import_module
from types import ModuleType, SimpleNamespace
import traceback
from itertools import chain

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
  skip_internal_package = kwargs.setdefault("skip_internal_package", True)
  path_from_package = kwargs.setdefault("path_from_package", "")
  path = package_path if not path_from_package else join(package_path, path_from_package)
  if extension is not None :
    extension = extension.lower()
  try :
    path_from_package_dot_notation = '.'.join(PurePath(path_from_package).parts)
    dict = {}
    entries = scandir(path)
    for entry in entries :
      entry_name = entry.name
      if isdir(entry) :
        kwargs["path_from_package"] = join(path_from_package, entry_name)
        if not skip_internal_package or not isfile(join(package_path, kwargs["path_from_package"], package_file)) :
          dict.update(all_files_in_path(package_path, **kwargs))
      elif entry_name not in exclude_files :
        file_name, file_extension = splitext(entry_name)
        if extension is None or file_extension.lower() == extension :
          dict['.'.join(filter(None, [path_from_package_dot_notation, file_name]))] = [file_name]
  except Exception :
    for entry in args :
      print (f"arg = {entry}")
    for key, value in kwargs.items():
      print (f"kwargs = <{key} : {value}>")
    print(traceback.format_exc())
  print(dict)
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
      self.__modules = set(modules)
      self.__class_to_module = {}
      for module, classlist in import_structure.items():
        for class_name in classlist:
          self.__class_to_module[class_name] = module
      # Needed for autocompletion in an IDE
      self.__all__ = list(modules) + list(chain(*classes))
      self.__file__ = package_file
#      self.__spec__ = package_spec
      self.__path__ = [package_dir]
      self.__objects = {} if extra_objects is None else extra_objects
      self.__package__ = package_name
      self.__import_structure = import_structure
      self.__allow_module_imports = True

    # Needed for autocompletion in an IDE
    def __dir__(self) :
      result = super().__dir__()
      # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
      # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
      for attr in self.__all__:
        if attr not in result :
          result.append(attr)
      return result

    def __getattr__(self, name: str) :
      if name in self.__objects:
        return self.__objects[name]
      if self.__allow_module_imports and name in self.__modules:
        value = self.__get_module(name)
        setattr(self, name, value)
        return value
      if name in self.__class_to_module.keys():
        value = getattr(self.__get_module(self.__class_to_module[name]), name)
        setattr(self, name, value)
        return value
      raise AttributeError(f"Package {self.__name__} has no module {name}")

    def __get_module(self, name: str) :
      try :
        return import_module("." + name, self.__name__)
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

    def allow_module_imports(enabled = True) :
      self.__allow_module_imports = enabled


def AutoLoad(name, file, spec, **kwargs) :
  #module_info = _getframe(1).f_globals
  #name = module_info["__name__"]
  #file = module_info["__file__"]
  #spec = module_info["__spec__"]
  module = LazyModule(name, file, spec = spec, **kwargs)
  modules[name] = module
  return module
