import json
import torch

try:
    from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    from transformers.utils.hub import cached_file
except ImportError:
    WEIGHTS_NAME = "pytorch_model.bin"
    CONFIG_NAME = "config.json"
    cached_file = None


def load_config_hf(model_name):
    if cached_file is None:
        raise ImportError("transformers is required to load config from HuggingFace hub")
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))


def load_state_dict_hf(model_name, device=None, dtype=None):
    if cached_file is None:
        raise ImportError("transformers is required to load state dict from HuggingFace hub")
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    return torch.load(resolved_archive_file, map_location=mapped_device)
