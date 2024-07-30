import sys

def module(cls):
    mod = __import__(cls.__module__)
    for name, method in inspect.getmembers(cls) :
      setattr(mod, name, method)
