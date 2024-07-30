#!/usr/bin/env python

import setuptools

setuptools.setup(
  name = 'stablediffusers',
  version = '0.1',
  description = 'Stable Diffusion library.',
  long_description = 'A convenience wrapper around the diffusers library for use with Stable Diffusion.',
  url ='https://github.com/jslegers/stablediffusers',
  author ='John Slegers',
  license = 'MIT',
  packages = ['stablediffusers'],
  install_requires =[
    'torch',
    'numba',
    'diffusers',
    'accelerate',
    'huggingface_hub',
    'diffusers',
    'transformers'
  ]
)
