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
        instance.__dict__[self.name] = (self, value)

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
    instance.__attr[self.name](*args, **kwargs)
    if not instance._Module_Attr__PROXY__activated :
      print(f"CALL --  instance.__attr[{self.name}]([{args}], {kwargs})")
  def __get__(self, instance, owner):
    print(instance._Module_Attr__PROXY__activated)
    print(f"GET --  instance.__attr[{self.name}]")
    if not instance._Module_Attr__PROXY__activated :
      return instance.__attr[self.name] if self.name in instance.__attr else None
  def __set__(self, instance, value):
    print(f"SET --  instance.__attr[{self.name}] = {value}")
    instance.__attr[self.name] = value

def create_module_proxy(name, attrs) :

  class Module_proxy(object):
    _Module_Attr__attr = {}
    attr_names = []
    attrs = []
    attrs_dict = {}
    _Module_Attr__PROXY__activated = False

    def get(self, key):
      return getattr(self, key)

    @classmethod
    def activate(cls) :
      Module_proxy._Module_Attr__PROXY__activated = True

  class Module_proxy_child(Module_proxy):
    @classmethod
    def setupattr(cls, name, parent):
      proxy = cls(name, parent)
      for attr in Module_proxy.attrs_dict :
        if attr == name :
          setattr(Module_proxy, attr, Module_proxy.attrs_dict[attr])
          delattr(Module_proxy_parent, attr)
      return proxy

    def __init__(self, name, parent) :
      self.__name = name
      self.__parent = parent

    def __getattr__(self, key):
      if name :
        self.activate()
        return None
      else :
        return None

  class Module_proxy_parent(Module_proxy):
    @classmethod
    def setup(cls, name, attrs = None):
      cls.__name = name
      proxy = Module_proxy_parent()
      if not attrs :
        return proxy
      if isinstance(attrs, str) :
        a = Module_Attr(attrs)
        setattr(Module_proxy_parent, attrs, a)
        Module_proxy.attrs.append(a)
        Module_proxy.attr_names.append(attrs)
        Module_proxy.attrs_dict[attrs] = a
        Module_proxy.attrs[-1] = Module_proxy_child.setupattr(name, proxy)
        return proxy
      for attr in attrs :
        a = Module_Attr(attr)
        setattr(Module_proxy_parent, attr, a)
        Module_proxy.attrs.append(a)
        Module_proxy.attr_names.append(attr)
        Module_proxy.attrs_dict[attr] = a
        Module_proxy.attrs[-1] = Module_proxy_child.setupattr(name, proxy)
      return proxy

    def __getitem__(self, key):
      return type(self).attrs[key]

  proxy = Module_proxy_parent.setup(name, attrs)
  #return Module_proxy()
  return proxy

PIL = create_module_proxy("PIL", ["Image", "ImageDraw", "ImageFont"])

Image, ImageDraw, ImageFont = PIL

cuda = create_module_proxy("torch.cuda", ["empty_cache", "ipc_collect"])

test4 = create_module_proxy("diffusers.utils", "logging")

Test = cuda.empty_cache

print(Image)
print(PIL.ImageDraw)

Image.ImageDraw = "Test"

print(cuda.empty_cache)
cuda.activate()
print(cuda.empty_cache)
print(Image.ImageDraw)
print(PIL.Image)
Image.activate()
print(ImageDraw)
print(PIL.Image)
print(ImageFont)
