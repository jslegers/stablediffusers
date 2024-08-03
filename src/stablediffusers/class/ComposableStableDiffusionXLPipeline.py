from stablediffusers.util import AutoLoad
import sys

module = AutoLoad(import_structure = {
  "torch" : ["bfloat16", "float16", "device"],
  "torch.cuda" : ["is_available", "ipc_collect", "empty_cache"],
  "numba.cuda" : ["select_device", "get_current_device"],
  "gc" : ["collect"],
  "accelerate" : ["init_empty_weights"],
  "huggingface_hub" : ["hf_hub_download", "snapshot_download"],
  "diffusers.models.model_loading_utils" : ["load_model_dict_into_meta"],
  "diffusers.utils" : ["logging"],
  "diffusers" : ["StableDiffusionXLPipeline", "UNet2DConditionModel", "AutoencoderKL"],
  "transformers" : ["CLIPTextModel", "CLIPTextModelWithProjection"],
  "sd_embed.embedding_funcs" : ["get_weighted_text_embeddings_sdxl"],
  "PIL" : ["Image", "ImageDraw", "ImageFont"],
  "os.path" : ["join"]
}).load('device').load('logging').get_logger(__name__)

# bfloat16, float16, dev, Generator = util.from_module('torch').load('bfloat16', 'float16', 'device', 'Generator')
"""
from torch import bfloat16, float16, device as dev, Generator
from torch.cuda import is_available, ipc_collect, empty_cache
from numba.cuda import select_device, get_current_device
from gc import collect
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers.models.model_loading_utils import load_model_dict_into_meta
from diffusers.utils import logging
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl
from PIL import Image, ImageDraw, ImageFont
from os.path import join
"""
import cv2
#dev = module.load('device')

class ComposableStableDiffusionXLPipeline:

  test = None

#  logger = module.load('logging').get_logger(__name__)
