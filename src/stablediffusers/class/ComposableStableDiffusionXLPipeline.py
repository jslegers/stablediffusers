from stablediffusers.util import module

cv2 = module("cv2")
Image, ImageDraw, ImageFont = module("PIL", ["Image", "ImageDraw", "ImageFont"])
join = module(module("os", "path"), "join")

get_weighted_text_embeddings_sdxl = module("sd_embed.embedding_funcs", "get_weighted_text_embeddings_sdxl")

collect = module("gc", "collect")
empty_cache, ipc_collect, set_device = module(module("torch", "cuda"), ["empty_cache", "ipc_collect", "set_device"])
init_empty_weights = module("accelerate", "init_empty_weights")
load_model_dict_into_meta = module("diffusers.models.model_loading_utils", "load_model_dict_into_meta")

how_many_gpus = module("torch", "cuda").device_count()
cuda_is_available = module("torch", "cuda").is_available()

logger = module("diffusers.utils", "logging").get_logger(__name__)
logger.setLevel("ERROR")

tqdm = module("diffusers.utils", "logging").tqdm

default = {
  "SDXL" : {
    "model" : module("diffusers", ["StableDiffusionXLPipeline", "FluxPipeline"]),
    "base_model" : "stabilityai/stable-diffusion-xl-base-1.0",
    "merging" : {
      "text_encoder" : {
        "model" : module("transformers", "CLIPTextModel"),
        "alpha" : 0.5,
        "skip_config_check" : True
      },
      "text_encoder_2" : {
        "model" : module("transformers", "CLIPTextModelWithProjection"),
        "alpha" : 0.5,
        "skip_config_check" : True
      },
      "unet" : {
        "model" : module("diffusers", "UNet2DConditionModel"),
        "alpha" : 0.5,
        "skip_config_check" : True
      },
      "vae" : {
        "model" : module("diffusers", "AutoencoderKL"),
        "alpha" : 0.5,
        "skip_config_check" : True
      }
    }
  }, "FLUX" : {
    "model" : module("diffusers", "FluxPipeline"),
    "base_model" : "black-forest-labs/FLUX.1-schnell",
    "tokenizer_2" : module("transformers", "T5TokenizerFast"),
    "merging" : {
      "text_encoder" : {
        "model" : module("transformers", "CLIPTextModel"),
        "alpha" : 0.5,
        "skip_config_check" : True
      },
      "text_encoder_2" : {
        "model" : module("transformers", "T5EncoderModel"),
        "alpha" : 0.5,
        "skip_config_check" : True
      },
      "vae" : {
        "model" : module("diffusers", "AutoencoderKL"),
        "alpha" : 0.5,
        "skip_config_check" : True
      }
    }
  }
}

default.update({
  "inference" : {
    "torch_dtype" : module("torch").float16,
    "variant" : "fp16",
    "use_safetensors" : True
  } if cuda_is_available else {
    "torch_dtype" : module("torch").bfloat16,
    "variant" : "bf16",
    "use_safetensors" : True
  }
})

def compare_configs(config_a, config_b, skip_keys):
  mismatched_keys = set()
  for key, value_a in config_a.items():
    if key in skip_keys:
      continue
    value_b = config_b.get(key)
    if value_a != value_b:
      mismatched_keys.add(key)
  return mismatched_keys

class ComposableStableDiffusionXLPipeline:

  device = module("torch").device("cuda" if cuda_is_available else "cpu")
  generator = module("torch").Generator(device = device)

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
    for _ in range(how_many_gpus):
      set_device(_)
      empty_cache()

  @classmethod
  def load_model(cls, *args, **kwargs):
    path, *_ = list(args) + [None]
    pipeline = kwargs.pop("pipeline", "FLUX")
    model = default[pipeline]["model"]
    path = path if path else default[pipeline]["base_model"]
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
        logger.info(f"Loading model {name} from memory")
        return cls
    logger.info(f"Loading model {name} from {path}")
    inference = default["inference"].copy()
    if pipeline == "FLUX" :
      tokenizer_2 = default[pipeline]["tokenizer_2"].from_pretrained(
        "johnslegers/FLUX.1-schnell-tokenizer_2",
        subfolder = "tokenizer_2",
        add_prefix_space = None
      )
      inference.setdefault("tokenizer_2", tokenizer_2)
    try :
      pipeline = default[pipeline]["merging"][name]["model"].from_pretrained(path, **inference)
    except :
      logger.info("Using default variant instead")
      inference.pop("variant")
      dtype = inference.pop("torch_dtype")
      pipeline = model.from_pretrained(path, **kwargs, **inference).to(dtype = dtype)
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
      logger.info(f"Unloading model '{name}'")
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
  def combine_tuples_into_dict(cls, *args, **kwargs):
    tuple1, tuple2, *_ = list(args) + [()] * 2
    if len(tuple1) == len(tuple2):
      return {tuple1[i] : tuple2[i] for i, _ in enumerate(tuple2)}
    raise Exception(f"{tuple1} is not the same size as {tuple2}")

  @classmethod
  def prompt_fix(cls, *args, **kwargs):
    prompt, *_ = list(args) + [kwargs.pop("prompt", None)]
    return cls.combine_tuples_into_dict((
      "prompt_embeds",
      "prompt_neg_embeds",
      "pooled_prompt_embeds",
      "negative_pooled_prompt_embeds"
    ), get_weighted_text_embeddings_sdxl(cls.current[2], prompt = ', '.join(filter(None, (
      prompt,
      kwargs.pop("prompt_2", None)
    ))), neg_prompt = ', '.join(filter(None, (
      kwargs.pop("negative_prompt", None),
      kwargs.pop("negative_prompt_2", None)
    )))))

  @classmethod
  def compose(cls, *args, **kwargs):
    pipeline = kwargs.pop("pipeline", "FLUX")
    name = kwargs.setdefault("name", None)
    if name is None :
      raise Exception ("Composite models must have a name")
    if cls.__get_model_from_store(name, by_name = True) is not None :
      raise Exception ("Models must have a unique name")
    path, *_ = list(args) + [None]
    if path is None :
      path = default[pipeline]["model"] if cls.current is None else cls.current[0]
    model = cls.__get_model_from_store(path)
    if model is not None :
      model = model[2]
      if pipeline == "SDXL" :
        kwargs.pop("unet", model.unet)
      kwargs.setdefault("text_encoder", model.text_encoder)
      kwargs.setdefault("text_encoder_2", model.text_encoder_2)
      kwargs.setdefault("vae", model.vae)
    return cls.load_model(path, skip_load_from_memory = True, **kwargs)

  @classmethod
  def wrap_text(cls, text, max_width, font):
    lines = []
    current_line = ""
    words = text.split(" ")
    for word in words:
      test_line = current_line + word + " "
      line_width = font.getlength(test_line)
      if line_width <= max_width:
        current_line = test_line
      else:
        lines.append(current_line[:-1])
        current_line = word + " "
    lines.append(current_line[:-1])
    return '\n'.join(lines)

  @classmethod
  def image_grid(cls, imgs, rows = 1, cols = 1, prompt = ""):
    assert len(imgs) == rows*cols
    text_margin = 40
    w, h = imgs[0].size
    prompt_height = h * rows // 2 - (2 * text_margin)
    prompt_width = cols*w - (2 * text_margin)
    grid = Image.new('RGB', size=(cols*w, rows*h + prompt_height))
    grid_w, grid_h = grid.size
    grid.paste((255,255,255, 255), (0, 0, grid_w, grid_h))
    draw = ImageDraw.Draw(grid)
    # requires a newer version of pillow
    # use a truetype font
    font_path = join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font_size = 30
    font = ImageFont.truetype(font_path, font_size)
    draw.text((text_margin, text_margin), cls.wrap_text(prompt, prompt_width, font), font = font, fill=(0,0,0, 255))
    for i, img in enumerate(imgs):
      grid.paste(img, box=(i%cols*w, prompt_height + (i//cols*h)))
    return grid

  @classmethod
  def __load_component_from_config(cls, config, **kwargs):
    name = kwargs.setdefault("name", "unet")
    pipeline = kwargs.setdefault("pipeline", "FLUX")
    model = default[pipeline]["merging"][name]["model"]
    return model(config) if "text_encoder" in name else model.from_config(config)

  @classmethod
  def __get_component(cls, path, **kwargs):
    name = kwargs.setdefault("name", "unet")
    pipeline = kwargs.setdefault("pipeline", "FLUX")
    model = default[pipeline]["merging"][name]["model"]
    inference = default["inference"]
    if path in cls.path :
      return getattr(cls.path[path][2], name)
    else :
      inference = inference.copy()
      try :
        return model.from_pretrained(path, **inference, **{
          "subfolder" : name
        })
      except :
        logger.info("Using default variant instead")
        inference.pop("variant")
        dtype = inference.pop("torch_dtype")
        return model.from_pretrained(path, **inference, **{
          "subfolder" : name
        }).to(dtype = dtype)

  @classmethod
  def merge(cls, model_a_name, model_b_name, **kwargs):
    pipeline = kwargs.setdefault("pipeline", "FLUX")
    model = kwargs.setdefault("model", "unet")
    alpha = kwargs.setdefault("alpha", default[pipeline]["merging"][model]["alpha"])
    skip_config_check = kwargs.setdefault("skip_config_check", default[pipeline]["merging"][model]["skip_config_check"])
    dtype = kwargs.setdefault("torch_dtype", default["inference"]["torch_dtype"])

    model_a = cls.__get_component(model_a_name, name = model)
    model_b = cls.__get_component(model_b_name, name = model)

    logger.info(f"Verifying {model} model compatibility...")
    keys_to_skip = {"_diffusers_version", "_name_or_path", "_use_default_values"}

    if not skip_config_check:
      # Compare configs
      mismatched_keys = compare_configs(model_a.config, model_b.config, keys_to_skip)

      if mismatched_keys:
        logger.error(f"{model.capitalize()} models have different configurations. Mismatched keys:")
        for key in mismatched_keys:
          logger.error(key)
        raise ValueError(f"{model.capitalize()} models cannot be merged due to configuration differences.")

      logger.info(f"{model.capitalize()} models are compatible.")

    merged_state_dict = {}

    for key in tqdm(model_a.state_dict().keys(), desc=f"Merging {model} models"):
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

    logger.info(f"Creating merged {model} model...")
    with init_empty_weights():
      merged_model = cls.__load_component_from_config(model_a.config, name = model, pipeline = pipeline)

    load_model_dict_into_meta(merged_model, merged_state_dict, device = cls.device, dtype = dtype)

    return merged_model
