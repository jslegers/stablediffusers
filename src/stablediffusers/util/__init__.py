from sys import modules
from os import scandir
from os.path import join, dirname, splitext, isfile
from pathlib import PurePath
from importlib import import_module
from types import ModuleType, SimpleNamespace
from itertools import chain

def snake_to_camel(word) :
  return ''.join(x.capitalize() or '_' for x in word.split('_'))

def camel_to_snake(s) :
  return ''.join(['_'+c.lower() if c.isupper() else c for c in s]).lstrip('_')

def all_files_in_path(
  package_path,
  path_from_package = None,
  skip_internal_package = False,
  extension = None,
  exclude_files = []
) :
  package_file = "__init__.py"
  path = join(package_path, path_from_package)
  if extension is not None :
    extension = extension.lower()
  path_from_package_dot_notation = '.'.join(PurePath.parts(path_from_package))
  dict = {}
  entries = scandir(path)
  for entry in entries :
    if entry.is_dir() :
      path_from_package = join(path_from_package, entry.name)
      if not skip_internal_package or not isfile(join(path_from_package, package_file)) :
        dict.update(all_files_in_path(
          package_path,
          join(path_from_package, entry.name),
          extension,
          skip_internal_package
        ))
    elif entry.name not in exclude_files :
      file_name, file_extension = splitext(entry.name)
      if file_extension.lower() == extension :
        dict[f"{path_from_package_dot_notation}.{file_name}"] = [file_name]
  return dict

class LazyModule(ModuleType) :
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(
      self,
      package_name,
      package_file,
      package_spec = None,
      import_structure = None,
      extra_objects = None
    ) :
      super().__init__(package_name)
      package_dir = dirname(package_file)
      if import_structure is None or isinstance(import_structure, str) :
        import_structure = all_files_in_path(
          package_dir,
          import_structure,
          extension = ".py",
          exclude_files = ["__init__.py"],
          skip_internal_package = True
        )
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
      self.__spec__ = package_spec
      self.__path__ = [package_dir]
      self.__objects = {} if extra_objects is None else extra_objects
      self.__name = package_name
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
        self.__name,
        self.__file__,
        self.__import_structure
      ))

    def allow_module_imports(enabled = True) :
      self.__allow_module_imports = enabled


AutoLoad(*args, **kwargs) :
  module = LazyModule(*arg, **kwargs)
  modules[package_name] = module
  return module
