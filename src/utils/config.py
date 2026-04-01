import os
import yaml


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(path):
    return load_yaml(path)


def get_device(prefer_mps=True):
    import torch
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
