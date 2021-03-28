import random
from addict import Dict
from PIL import Image, ImageOps, ImageEnhance


class Transform:
    """
    Base class for transform operations.
    """
    def __init__(self, config):
        self.config = config

    def __call__(self, data, raw_input_idx):
        """
        Args:
            data (list): List of list. The inner list contains the different sources, Images, for one instance.
                  The sequence of the inner list follows the given source_dirs.
            raw_input_idx (int): The index of the rgb input. -1 means rgb not existed.
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
        self.apply_prob = config.apply_prob
        self.value_range = config.value_range
        self.gen_rand_value = lambda: random.uniform(self.value_range[0], self.value_range[1])

    def __call__(self, data, _):
        prob = random.random()
        value = self.gen_rand_value()
        augmented = []
        for img in data:
            if prob < self.apply_prob:
                rotated = img.rotate(value)
                augmented.append(rotated)
            else:
                augmented.append(img)
        return augmented


class ShearX(Transform):
    def __init__(self, config):
        super(ShearX, self).__init__(config)
        self.apply_prob = config.apply_prob
        self.value_range = config.value_range
        self.gen_rand_value = lambda: random.uniform(self.value_range[0], self.value_range[1])

    def __call__(self, data, _):
        prob = random.random()
        value = self.gen_rand_value()
        augmented = []
        for i, img in enumerate(data):
            if prob < self.apply_prob:
                augmented.append(img.transform(img.size, Image.AFFINE, (1, value, 0, 0, 1, 0)))
            else:
                augmented.append(img)
        return augmented


class ShearY(Transform):
    def __init__(self, config):
        super(ShearY, self).__init__(config)
        self.apply_prob = config.apply_prob
        self.value_range = config.value_range
        self.gen_rand_value = lambda: random.uniform(self.value_range[0], self.value_range[1])

    def __call__(self, data, raw_input_idx):
        prob = random.random()
        value = self.gen_rand_value()
        augmented = []
        for i, img in enumerate(data):
            if prob < self.apply_prob:
                augmented.append(img.transform(img.size, Image.AFFINE, (1, 0, value, 0, 1, 0)))
            else:
                augmented.append(img)
        return augmented


class TranslateX(Transform):
    def __init__(self, config):
        super(TranslateX, self).__init__(config)
        self.apply_prob = config.apply_prob
        self.value_range = config.value_range
        self.gen_rand_value = lambda: random.uniform(self.value_range[0], self.value_range[1])

    def __call__(self, data, raw_input_idx):
        prob = random.random()
        value = self.gen_rand_value()
        augmented = []
        for i, img in enumerate(data):
            if prob < self.apply_prob:
                pixel_value = value * img.size[0]
                augmented.append(img.transform(img.size, Image.AFFINE, (1, 0, pixel_value, 0, 1, 0)))
            else:
                augmented.append(img)
        return augmented


class TranslateY(Transform):
    def __init__(self, config):
        super(TranslateY, self).__init__(config)
        self.apply_prob = config.apply_prob
        self.value_range = config.value_range
        self.gen_rand_value = lambda: random.uniform(self.value_range[0], self.value_range[1])

    def __call__(self, data, raw_input_idx):
        prob = random.random()
        value = self.gen_rand_value()
        augmented = []
        for i, img in enumerate(data):
            if prob < self.apply_prob:
                pixel_value = value * img.size[0]
                augmented.append(img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixel_value)))
            else:
                augmented.append(img)
        return augmented


class AutoContrast(Transform):
    def __init__(self, config):
        super(AutoContrast, self).__init__(config)
        self.apply_all = config.apply_all
        self.apply_prob = config.apply_prob

    def __call__(self, data, raw_input_idx):
        prob = random.random()
        augmented = []
        for i, img in enumerate(data):
            if (self.apply_all or i == raw_input_idx) and prob < self.apply_prob:
                try:
                    augmented.append(ImageOps.autocontrast(img))
                except OSError:
                    augmented.append(img)
            else:
                augmented.append(img)
        return augmented


class Invert(Transform):
    def __init__(self, config):
        super(Invert, self).__init__(config)
        self.apply_all = config.apply_all
        self.apply_prob = config.apply_prob

    def __call__(self, data, raw_input_idx):
        prob = random.random()
        augmented = []
        for i, img in enumerate(data):
            if (self.apply_all or i == raw_input_idx) and prob < self.apply_prob:
                try:
                    augmented.append(ImageOps.invert(img))
                except OSError:
                    augmented.append(img)
            else:
                augmented.append(img)
        return augmented


class Equalize(Transform):
    def __init__(self, config):
        super(Equalize, self).__init__(config)
        self.apply_all = config.apply_all
        self.apply_prob = config.apply_prob

    def __call__(self, data, raw_input_idx):
        prob = random.random()
        augmented = []
        for i, img in enumerate(data):
            if (self.apply_all or i == raw_input_idx) and prob < self.apply_prob:
                try:
                    augmented.append(ImageOps.equalize(img))
                except OSError:
                    augmented.append(img)
            else:
                augmented.append(img)
        return augmented


class Mirror(Transform):
    def __init__(self, config):
        super(Mirror, self).__init__(config)
        self.apply_prob = config.apply_prob

    def __call__(self, data, raw_input_idx):
        prob = random.random()
        augmented = []
        for i, img in enumerate(data):
            if prob < self.apply_prob:
                augmented.append(ImageOps.mirror(img))
            else:
                augmented.append(img)
        return augmented


class Solarize(Transform):
    def __init__(self, config):
        super(Solarize, self).__init__(config)
        self.apply_prob = config.apply_prob
        self.apply_all = config.apply_all
        self.value_range = config.value_range
        self.gen_rand_value = lambda: random.uniform(self.value_range[0], self.value_range[1])

    def __call__(self, data, raw_input_idx):
        prob = random.random()
        value = self.gen_rand_value()
        augmented = []
        for i, img in enumerate(data):
            if (self.apply_all or i == raw_input_idx) and prob < self.apply_prob:
                try:
                    augmented.append(ImageOps.solarize(img, value))
                except OSError:
                    augmented.append(img)
            else:
                augmented.append(img)
        return augmented


class Posterize(Transform):
    def __init__(self, config):
        super(Posterize, self).__init__(config)
        self.apply_prob = config.apply_prob
        self.apply_all = config.apply_all
        self.value_range = config.value_range
        self.gen_rand_value = lambda: random.randint(self.value_range[0], self.value_range[1])

    def __call__(self, data, raw_input_idx):
        prob = random.random()
        value = self.gen_rand_value()
        augmented = []
        for i, img in enumerate(data):
            if (self.apply_all or i == raw_input_idx) and prob < self.apply_prob:
                try:
                    augmented.append(ImageOps.posterize(img, value))
                except OSError:
                    augmented.append(img)
            else:
                augmented.append(img)
        return augmented


class Contrast(Transform):
    def __init__(self, config):
        super(Contrast, self).__init__(config)
        self.apply_prob = config.apply_prob
        self.apply_all = config.apply_all
        self.value_range = config.value_range
        self.gen_rand_value = lambda: random.uniform(self.value_range[0], self.value_range[1])

    def __call__(self, data, raw_input_idx):
        prob = random.random()
        value = self.gen_rand_value()
        augmented = []
        for i, img in enumerate(data):
            if (self.apply_all or i == raw_input_idx) and prob < self.apply_prob:
                try:
                    augmented.append(ImageEnhance.Contrast(img).enhance(value))
                except OSError:
                    augmented.append(img)
            else:
                augmented.append(img)
        return augmented


class Color(Transform):
    def __init__(self, config):
        super(Color, self).__init__(config)
        self.apply_prob = config.apply_prob
        self.apply_all = config.apply_all
        self.value_range = config.value_range
        self.gen_rand_value = lambda: random.uniform(self.value_range[0], self.value_range[1])

    def __call__(self, data, raw_input_idx):
        prob = random.random()
        value = self.gen_rand_value()
        augmented = []
        for i, img in enumerate(data):
            if (self.apply_all or i == raw_input_idx) and prob < self.apply_prob:
                try:
                    augmented.append(ImageEnhance.Color(img).enhance(value))
                except OSError:
                    augmented.append(img)
            else:
                augmented.append(img)
        return augmented


class Brightness(Transform):
    def __init__(self, config):
        super(Brightness, self).__init__(config)
        self.apply_prob = config.apply_prob
        self.apply_all = config.apply_all
        self.value_range = config.value_range
        self.gen_rand_value = lambda: random.uniform(self.value_range[0], self.value_range[1])

    def __call__(self, data, raw_input_idx):
        prob = random.random()
        value = self.gen_rand_value()
        augmented = []
        for i, img in enumerate(data):
            if (self.apply_all or i == raw_input_idx) and prob < self.apply_prob:
                try:
                    augmented.append(ImageEnhance.Brightness(img).enhance(value))
                except OSError:
                    augmented.append(img)
            else:
                augmented.append(img)
        return augmented


class Sharpness(Transform):
    def __init__(self, config):
        super(Sharpness, self).__init__(config)
        self.apply_prob = config.apply_prob
        self.apply_all = config.apply_all
        self.value_range = config.value_range
        self.gen_rand_value = lambda: random.uniform(self.value_range[0], self.value_range[1])

    def __call__(self, data, raw_input_idx):
        prob = random.random()
        value = self.gen_rand_value()
        augmented = []
        for i, img in enumerate(data):
            if (self.apply_all or i == raw_input_idx) and prob < self.apply_prob:
                try:
                    augmented.append(ImageEnhance.Sharpness(img).enhance(value))
                except OSError:
                    augmented.append(img)
            else:
                augmented.append(img)
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
        'ShearY': ShearY,
        'TranslateX': TranslateX,
        'TranslateY': TranslateY,
        'AutoContrast': AutoContrast,
        'Invert': Invert,
        'Equalize': Equalize,
        'Mirror': Mirror,
        'Solarize': Solarize,
        'Posterize': Posterize,
        'Contrast': Contrast,
        'Color': Color,
        'Brightness': Brightness,
        'Sharpness': Sharpness,
    }
    return transform_registry[config.type](config)
