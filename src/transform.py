import random
from addict import Dict
from PIL import Image, ImageOps, ImageEnhance


class Transform:
    """
    Base class for transform operations.
    """
    def __init__(self, config):
        self.config = config

    def __call__(self, data):
        """
        Args:
            data (list): List of list. The inner list contains the different sources, Images, for one instance.
                  The sequence of the inner list follows the given source_dirs.
        Returns:
            dict: Dict of {src_name: Image}, where Images are augmented.
        """
        raise RuntimeError("Illegal call to base class.")


class Rotate(Transform):
    """
    Rotate the image and fill the gap with specific color.
    """
    def __init__(self, config):
        super(Rotate, self).__init__(config)
        self.fill_color = config.fill_color
        self.value_range = config.value_range
        self.gen_rand_value = lambda: random.uniform(self.value_range[0], self.value_range[1])

    def __call__(self, data):
        value = self.gen_rand_value()
        augmented = []
        for img in data:
            rotated = img.convert("RGBA").rotate(value)
            rotated = Image.composite(rotated,
                                      Image.new("RGBA", rotated.size, (self.fill_color,) * 4),
                                      rotated).convert(img.mode)
            augmented.append(rotated)
        return augmented


class ShearX(Transform):
    def __init__(self, config):
        super(ShearX).__init__(config)
        self.fill_color = config.fill_color
        self.apply_all = config.apply_all

    def __call__(self, data, degree):
        """
        Args:
            data (dict): Dict of {src_name: Image}
        Returns:
            dict: Dict of {src_name: Image}, where Images are augmented.
        """
        augmented = {}
        for src_name, img in data.items():
            if self.apply_all or src_name == 'rgb':
                rotated = img.convert("RGBA").rotate(degree)
                rotated = Image.composite(rotated,
                                          Image.new("RGBA", rotated.size, (self.fill_color,) * 4),
                                          rotated).convert(img.mode)
                augmented[src_name] = rotated
        return augmented


def build_transform(config):
    if not isinstance(config, Dict):
        raise TypeError(f'config must be a addict.Dict, but got {type(config)}')
    if 'type' not in config:
        raise KeyError(
                f'`config` must contain the key "type", but got {config}')
    transform_registry = {
        'Rotate': Rotate,
        'ShearX': ShearX,
    }
    return transform_registry[config.type](config)
