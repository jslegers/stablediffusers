import sys

def module(cls):
    mod = __import__(cls.__module__)
    mod.root = cls
