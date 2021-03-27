import cv2
from addict import Dict
TRANSFORM_REGISTRY = {}


def register(transform, registry):
    registry[transform.__name__] = transform


def build_transform(config):
    if not isinstance(config, Dict):
        raise TypeError(f'config must be a addict.Dict, but got {type(config)}')
    if 'type' not in config:
        raise KeyError(
                f'`config` must contain the key "type", but got {config}')
    return TRANSFORM_REGISTRY[config.type](config)


