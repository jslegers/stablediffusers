import abc

class Property:
    "Emulate PyProperty_Type() in Objects/descrobject.c"

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc
        self._name = ''

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError(
                f'property {self._name!r} of {type(obj).__name__!r} object has no getter'
             )
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError(
                f'property {self._name!r} of {type(obj).__name__!r} object has no setter'
             )
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError(
                f'property {self._name!r} of {type(obj).__name__!r} object has no deleter'
             )
        self.fdel(obj)

    def getter(self, fget):
        prop = type(self)(fget, self.fset, self.fdel, self.__doc__)
        prop._name = self._name
        return prop

    def setter(self, fset):
        prop = type(self)(self.fget, fset, self.fdel, self.__doc__)
        prop._name = self._name
        return prop

    def deleter(self, fdel):
        prop = type(self)(self.fget, self.fset, fdel, self.__doc__)
        prop._name = self._name
        return prop



class Attr:
    def __init__(self, value):
        self.name = value
        print(f"INIT --  self.name = {value}")
    def __call__(self, instance, *args, **kwargs):
        instance.__dict__[self.name](*args, **kwargs)
        print(f"CALL --  instance.__dict__[{self.name}]([{args}], {kwargs})")
    def __get__(self, instance, owner):
        print(f"GET --  instance.__dict__[{self.name}]")
        return instance.__dict__[self.name]
    def __set__(self, instance, value):
        print(f"SET --  instance.__dict__[{self.name}] = {value}")
        instance.__dict__[self.name] = value

class MyClass():

    #name = Attr("name")
    #home = Attr("home")
    #profession = Attr("profession")

    def __init__(self):
        self.name = "John"
        self.home = "Bilzen"
        self.profession = "Dev"


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
  def __set__(self, instance, value):
    print(f"SET --  instance.__dict__[{self.name}] = {value}")
    if not instance._Module_Attr__PROXY__activated :
      instance.__module_proxy[self.name] = value
    else :
      setattr(instance.__module, self.name, value)

from importlib import import_module, util
import sys

def get_module_from_code(code):
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

def get_mod(fullname, attrs = None):
  if not attrs :
    code = f"from {fullname} import *"
    return get_module_from_code(code)
  if isinstance(attrs, str) :
    code = f"from {fullname} import {attrs}"
    return getattr(get_module_from_code(code), attrs)
  code = f"from {fullname} import {', '.join(attrs)}"
  return (getattr(get_module_from_code(code), attr) for attr in attrs)

import types

def create_module_proxy(name, attrs) :

  class Module_proxy(object):
    attr_names = []
    attrs = []
    attrs_dict = {}
    _Module_Attr__PROXY__activated = False
    _Module_Attr__module_proxy = {}
    _Module_Attr__module = None
    MODULY_PROXY_name = ''

    @classmethod
    def _Module_Attr__PROXY__activate(cls) :
      if not Module_proxy._Module_Attr__PROXY__activated :
        Module_proxy._Module_Attr__PROXY__activated = True
        print("ACTIVATE")
        mod = get_mod(cls.MODULY_PROXY_name, cls.attr_names)
        Module_proxy._Module_Attr__module = mod
        Module_proxy.attrs = []
        for key in Module_proxy.attr_names :
          attrval = next(mod)
          if callable(attrval) :
            def q(cls, *args, **kwargs) :
              return attrval(*args, **kwargs)
          else :
            q = attrval
          del Module_proxy.attrs_dict[key]
          Module_proxy.attrs_dict[key] = q
          Module_proxy.attrs.append(q)
          setattr(Module_proxy_parent, key, q)
        print(Module_proxy._Module_Attr__module)

    def __init__(self, name) :
      Module_proxy.MODULY_PROXY_name = name

  class Module_proxy_child(Module_proxy):
    @classmethod

    def setupattr(cls, name, parent):
      proxy = Module_proxy_child(name, parent)
      return proxy

    def __init__(self, name, parent) :
      self.MODULY_PROXY_name = name
      self.__parent = parent

    def __getattr__(self, key):
      self._Module_Attr__PROXY__activate()
      return getattr(Module_proxy.attrs_dict[self.MODULY_PROXY_name], self.MODULY_PROXY_name)

    def __str__(self):
      self._Module_Attr__PROXY__activate()
      return str(Module_proxy.attrs_dict[self.MODULY_PROXY_name])

    def __call__(self, *args, **kwargs):
      self._Module_Attr__PROXY__activate()
      return Module_proxy.attrs_dict[self.MODULY_PROXY_name](*args, **kwargs)


  class Module_proxy_parent(Module_proxy):

    @classmethod
    def setup(cls, name, attrs = None):
      proxy = Module_proxy_parent(name)
      if not attrs :
        return proxy
      if isinstance(attrs, str) :
        a = Module_Attr(attrs)
        setattr(Module_proxy_parent, attrs, a)
        Module_proxy.attrs.append(a)
        Module_proxy.attr_names.append(attrs)
        Module_proxy.attrs_dict[attrs] = a
        child = Module_proxy_child.setupattr(attrs, proxy)
        Module_proxy.attrs[-1] = child
        Module_proxy._Module_Attr__module_proxy[attrs] = child
        return proxy
      for attr in attrs :
        a = Module_Attr(attr)
        setattr(Module_proxy_parent, attr, a)
        Module_proxy.attrs.append(a)
        Module_proxy.attr_names.append(attr)
        Module_proxy.attrs_dict[attr] = a
        child = Module_proxy_child.setupattr(attr, proxy)
        Module_proxy.attrs[-1] = child
        Module_proxy._Module_Attr__module_proxy[attr] = child
      return proxy

    def __getattr__(self, key):
      return Module_proxy._Module_Attr__module_proxy[key].value

    def __getitem__(self, key):
      return type(self).attrs[key]

  proxy = Module_proxy_parent.setup(name, attrs)
  return proxy

PIL = create_module_proxy("PIL", ["Image", "ImageDraw", "ImageFont"])

Image, ImageDraw, ImageFont = PIL

cuda = create_module_proxy("torch.cuda", ["empty_cache", "ipc_collect", "device_count"])

test4 = create_module_proxy("diffusers.utils", "logging")

Test = cuda.empty_cache

print(PIL.ImageDraw)
print(ImageDraw)
print(PIL.ImageDraw)

print(cuda.empty_cache)

print(cuda.empty_cache)
print(PIL.Image)

print(ImageDraw)
print(PIL.Image)
print(ImageFont)
print(cuda.device_count)
how_many_gpus = cuda.device_count()
for _ in range(how_many_gpus):
  cuda.set_device(_)
  cuda.empty_cache()
