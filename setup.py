#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
  name = "stablediffusers",
  version = "0.1",
  description = "Stable Diffusion library.",
  long_description = "A convenience wrapper around the diffusers library for use with Stable Diffusion.",
  url ="https://github.com/jslegers/stablediffusers",
  author ="John Slegers",
  license = "MIT",
  package_dir={"": "src"}
  packages=find_packages("src"),
  python_requires=">=3.8.0",
  dependency_links = ["https://github.com/xhinker/sd_embed/tarball/master#egg=sd_embed-1.240707.2"]
  install_requires =[
    "accelerate",
    "diffusers",
    "filelock",
    "huggingface_hub",
    "importlib_metadata"
    "numba",
    "numpy",
    "Pillow",
    "regex",
    "requests",
    "safetensors",
    "torch",
    "torchaudio",
    "torchvision"
    "transformers"
  ]
)
