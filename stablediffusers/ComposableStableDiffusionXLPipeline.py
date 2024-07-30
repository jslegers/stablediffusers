from torch import bfloat16, float16, device as dev
from torch.cuda import is_available, ipc_collect, empty_cache
from numba.cuda import select_device, get_current_device
from gc import collect
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers.models.model_loading_utils import load_model_dict_into_meta
from diffusers.utils import logging
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTextModelWithProjection

class ComposableStableDiffusionXLPipeline:

  logger = logging.get_logger(__name__)
  logger.setLevel("ERROR")

  default = {
    "model" : "stabilityai/stable-diffusion-xl-base-1.0",
    "device" : "cuda" if is_available() else "cpu",
    "merging" : {
      "text_encoder" : {
        "model" : CLIPTextModel,
        "alpha" : 0.5,
        "skip_config_check" : True
      },
      "text_encoder_2" : {
        "model" : CLIPTextModelWithProjection,
        "alpha" : 0.5,
        "skip_config_check" : True
      },
      "unet" : {
        "model" : UNet2DConditionModel,
        "alpha" : 0.5,
        "skip_config_check" : True
      },
      "vae" : {
        "model" : AutoencoderKL,
        "alpha" : 0.5,
        "skip_config_check" : True
      }
    }
  }

  default.update({
    "inference" : {
      "torch_dtype" : float16,
      "variant" : "fp16",
      "use_safetensors" : True
    } if default["device"] == "cuda" else {
      "torch_dtype" : bfloat16,
      "variant" : "bf16",
      "use_safetensors" : True
    }
  })

  device = dev(default["device"])

  name = {}
  path = {}
  current = None

  @classmethod
  def __get_model_from_store(cls, *args, **kwargs):
    key, *_ = list(args) + [None]
    if key is not None :
      by_name = kwargs.pop("by_name", False)
      store = cls.name if by_name else cls.path
      if key in store :
        return store[key]
    return None

  @classmethod
  def __load_model_from_memory(cls, *args, **kwargs):
    path, *_ = list(args) + [None]
    name = kwargs.pop("name", None)
    return_name_if_different = kwargs.pop("return_name_if_different", False)
    by_path_must_match_by_name = kwargs.pop("by_path_must_match_by_name", False)
    by_name_if_by_path_failed = kwargs.pop("by_name_if_by_path_failed", False)
    return_current_if_not_found = kwargs.pop("return_current_if_not_found", False)
    if name is None or path == name :
      if path is not None :
        model_by_path = cls.__get_model_from_store(path)
        if model_by_path is not None :
          return model_by_path
      return cls.current if return_current_if_not_found else None
    model_by_path = cls.__get_model_from_store(path)
    model_by_name = cls.__get_model_from_store(name, by_name = True)
    if model_by_name is None :
      if model_by_path is not None :
        return model_by_path
      return cls.current if return_current_if_not_found else None
    if model_by_path is not None :
      if model_by_name is model_by_path :
        return model_by_name
      if not by_path_must_match_by_name :
        return model_by_name if return_name_if_different else model_by_path
      raise Exception(f"Model '{name}' does not match path '{path}'")
    if by_name_if_by_path_failed :
      return model_by_name
    raise Exception(f"Model '{name}' not found at '{path}'")


  @classmethod
  def flush(cls, *args, **kwargs):
    collect()
    empty_cache()
    ipc_collect()

  @classmethod
  def load_model(cls, *args, **kwargs):
    path, *_ = list(args) + [None]
    path = path if path else cls.default["model"]
    skip_load_from_memory = kwargs.pop("skip_load_from_memory", False)
    name = kwargs.pop("name", path)
    if not skip_load_from_memory :
      by_path = cls.__get_model_from_store(path)
      if by_path :
        by_name = cls.__get_model_from_store(name, by_name = True)
        cls.current = by_path
        if by_name is not by_path :
          cls.current[1].append(name)
          cls.name[name] = cls.current
        cls.logger.info(f"Loading model {name} from memory")
        return cls
    cls.logger.info(f"Loading model {name} from {path}")
    pipeline = StableDiffusionXLPipeline.from_pretrained(path, **kwargs, **cls.default["inference"])
    cls.name[name] = [None, [name], pipeline]
    cls.current = cls.name[name]
    if "unet" in kwargs or "text_encoder" in kwargs or "text_encoder_2" in kwargs or "vae" in kwargs :
      return cls
    cls.path[path] = cls.name[name]
    cls.name[name][0] = path
    return cls


  @classmethod
  def unload_model(cls, *args, **kwargs):
    model = cls.__load_model_from_memory(*args, **kwargs, **{
      "by_path_must_match_by_name" : True,
      "by_name_if_by_path_failed" : True,
      "return_current_if_not_found" : True,
    })
    if model is not None :
      name = kwargs["name"] if "name" in kwargs else model[0]
      cls.logger.info(f"Unloading model '{name}'")
      if model[0] is not None :
        del cls.path[model[0]]
      for name in model[1] :
        del cls.name[name]
      if cls.current is model :
        cls.current = list(cls.path.values())[-1] if len(cls.path) > 0 else None
      del model
      cls.flush()
      return cls
    raise Exception("Model not loaded")

  @classmethod
  def from_loaded(cls, *args, **kwargs):
    model = cls.__load_model_from_memory(*args, **kwargs, **{
      "by_path_must_match_by_name" : True,
      "by_name_if_by_path_failed" : True,
      "return_current_if_not_found" : True,
    })
    if model is not None :
      return model[2]
    raise Exception("No model available")

  @classmethod
  def compose(cls, *args, **kwargs):
    name = kwargs.setdefault("name", None)
    if name is None :
      raise Exception ("Composite models must have a name")
    if cls.__get_model_from_store(name, by_name = True) is not None :
      raise Exception ("Models must have a unique name")
    path, *_ = list(args) + [None]
    if path is None :
      path = cls.default["model"] if cls.current is None else cls.current[0]
    model = cls.__get_model_from_store(path)
    if model is not None :
      model = model[2]
      kwargs.setdefault("unet", model.unet)
      kwargs.setdefault("text_encoder", model.text_encoder)
      kwargs.setdefault("text_encoder_2", model.text_encoder_2)
      kwargs.setdefault("vae", model.vae)
    return cls.load_model(path, skip_load_from_memory = True, **kwargs)

  @classmethod
  def __load_component_from_config(cls, config, **kwargs):
    name = kwargs.setdefault("name", "unet")
    model = cls.default["merging"][name]["model"]
    return model(config) if "text_encoder" in name else model.from_config(config)

  @classmethod
  def __get_component(cls, path, **kwargs):
    name = kwargs.setdefault("name", "unet")
    if path in cls.path :
      return getattr(cls.path[path][2], name)
    else :
      return cls.default["merging"][name]["model"].from_pretrained(path, **cls.default["inference"], **{
        "subfolder" : name
      })

  @classmethod
  def __compare_configs(cls, config_a, config_b, skip_keys):
    mismatched_keys = set()
    for key, value_a in config_a.items():
      if key in skip_keys:
        continue
      value_b = config_b.get(key)
      if value_a != value_b:
        mismatched_keys.add(key)
    return mismatched_keys

  @classmethod
  def merge(cls, model_a_name, model_b_name, **kwargs):
    model = kwargs.setdefault("model", "unet")
    alpha = kwargs.setdefault("alpha", cls.default["merging"][model]["alpha"])
    skip_config_check = kwargs.setdefault("skip_config_check", cls.default["merging"][model]["skip_config_check"])
    torch_dtype = kwargs.setdefault("torch_dtype", cls.default["inference"]["torch_dtype"])

    model_a = cls.__get_component(model_a_name, name = model)
    model_b = cls.__get_component(model_b_name, name = model)

    cls.logger.info(f"Verifying {model} model compatibility...")
    keys_to_skip = {"_diffusers_version", "_name_or_path", "_use_default_values"}

    if not skip_config_check:
      # Compare configs
      mismatched_keys = cls.__compare_configs(model_a.config, model_b.config, keys_to_skip)

      if mismatched_keys:
        cls.logger.error(f"{model.capitalize()} models have different configurations. Mismatched keys:")
        for key in mismatched_keys:
          cls.logger.error(key)
        raise ValueError(f"{model.capitalize()} models cannot be merged due to configuration differences.")

      cls.logger.info(f"{model.capitalize()} models are compatible.")

    merged_state_dict = {}

    for key in logging.tqdm(model_a.state_dict().keys(), desc=f"Merging {model} models"):
      if key not in model_b.state_dict():
        raise ValueError(f"Key {key} not found in vae B")

      tensor_a = model_a.state_dict()[key].to(cls.device)
      tensor_b = model_b.state_dict()[key].to(cls.device)

      if tensor_a.shape != tensor_b.shape:
        raise ValueError(f"Shape mismatch for key {key}: A: {tensor_a.shape}, B: {tensor_b.shape}")

      merged_tensor = (1 - alpha) * tensor_a + alpha * tensor_b
      merged_state_dict[key] = merged_tensor.to(cls.device)

      # Clear GPU memory
      del tensor_a
      del tensor_b
      empty_cache()

    cls.logger.info(f"Creating merged {model} model...")
    with init_empty_weights():
      merged_model = cls.__load_component_from_config(model_a.config, name = model)

    load_model_dict_into_meta(merged_model, merged_state_dict, device = cls.device, dtype = cls.default["inference"]["torch_dtype"])

    return merged_model
