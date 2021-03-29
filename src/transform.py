import random
from addict import Dict
import numpy as np
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
        self.apply_all = config.apply_all[0]
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
        self.apply_all = config.apply_all[0]
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
        self.apply_all = config.apply_all[0]
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


class Flip(Transform):
    def __init__(self, config):
        super(Flip, self).__init__(config)
        self.apply_prob = config.apply_prob

    def __call__(self, data, raw_input_idx):
        prob = random.random()
        augmented = []
        for i, img in enumerate(data):
            if prob < self.apply_prob:
                augmented.append(ImageOps.flip(img))
            else:
                augmented.append(img)
        return augmented


class Solarize(Transform):
    def __init__(self, config):
        super(Solarize, self).__init__(config)
        self.apply_prob = config.apply_prob
        self.apply_all = config.apply_all[0]
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
        self.apply_all = config.apply_all[0]
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
        self.apply_all = config.apply_all[0]
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
                except OSError and ValueError:
                    augmented.append(img)
            else:
                augmented.append(img)
        return augmented


class Color(Transform):
    def __init__(self, config):
        super(Color, self).__init__(config)
        self.apply_prob = config.apply_prob
        self.apply_all = config.apply_all[0]
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
                except OSError and ValueError:
                    augmented.append(img)
            else:
                augmented.append(img)
        return augmented


class Brightness(Transform):
    def __init__(self, config):
        super(Brightness, self).__init__(config)
        self.apply_prob = config.apply_prob
        self.apply_all = config.apply_all[0]
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
                except OSError and ValueError:
                    augmented.append(img)
            else:
                augmented.append(img)
        return augmented


class Sharpness(Transform):
    def __init__(self, config):
        super(Sharpness, self).__init__(config)
        self.apply_prob = config.apply_prob
        self.apply_all = config.apply_all[0]
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
                except OSError and ValueError:
                    augmented.append(img)
            else:
                augmented.append(img)
        return augmented


class RandomResizedCrop(Transform):
    def __init__(self, config):
        super(RandomResizedCrop, self).__init__(config)
        self.scale = config.scale
        self.padding_mode = config.padding_mode

    def __call__(self, data, raw_input_idx):
        if len(data) == 0:
            return data
        if not isinstance(data[0], Image.Image):
            raise TypeError("Unexpected type {}".format(type(data[0])))

        augmented = []
        width, height = data[0].size

        resized_height, resized_width = self.get_resize_params((height, width), self.scale)
        pad_left, pad_top, pad_right, pad_bottom = self.get_pad_params(
            (height, width), (resized_height, resized_width), ratio=0.2)
        left, top, right, bottom = self.get_crop_params(
            (resized_height + pad_top + pad_bottom, resized_width + pad_left + pad_right), (height, width))

        for i, img in enumerate(data):
            if (width, height) != img.size:
                raise ValueError("Images are not in same size.")
            # resize
            img = img.resize((resized_width, resized_height))
            # padding
            img = self.pad(img, (pad_left, pad_top, pad_right, pad_bottom), self.padding_mode)
            # cropping
            img = img.crop((left, top, right, bottom))

            augmented.append(img)

        return augmented

    def get_resize_params(self, image_size, scale=(0.8, 1.2)):
        """Get resized image size.
        Args:
            image_size tuple (int, int): Image size of (height, width).
            scale tuple (float, float): (min, max) of resize ratio.
        Returns:
            tuple: Output image size of (height, width).
        """
        if self.scale[0] > self.scale[1]:
            raise ValueError("Scale ratio should be of kind (min, max)")

        ratio = random.randint(int(100 * scale[0]), int(100 * scale[1])) / 100.0
        height, width = image_size
        new_height = int(ratio * height)
        new_width = int(ratio * width)
        return new_height, new_width

    def get_pad_params(self, image_size, resized_image_size, ratio=0.2):
        """Get parameters for padding.
        Args:
            image_size tuple (int, int): Image size of (height, width).
            resized_image_size tuple (int, int): Resized image size of (height, width).
            ratio (float): Padding ratio of each side for original image.
        Returns:
            tuple: Params (pad_left, pad_top, pad_right, pad_bottom) for padding.
        """
        height, width = image_size
        resized_height, resized_width = resized_image_size
        if resized_height >= height * (1 + ratio):
            pad_h = 0
        else:
            pad_h = int((height * (1 + ratio) - resized_height) / 2)
        if resized_width >= width * (1 + ratio):
            pad_w = 0
        else:
            pad_w = int((width * (1 + ratio) - resized_width) / 2)
        return pad_w, pad_h, pad_w, pad_h

    def get_crop_params(self, input_size, output_size):
        """Get parameters for a random crop.
        Args:
            input_size tuple (int, int): Input image size of (height, width).
            output_size tuple (int, int): Output image size of (height, width).
        Returns:
            tuple: Params (left, top, right, bottom) for random crop.
        """
        h_in, w_in = input_size
        h_out, w_out = output_size
        if w_in + 1 < w_out or h_in + 1 < h_out:
            raise ValueError("Required crop size {} is larger than input image size {}"
                             .format((h_out, w_out), (h_in, w_in)))

        if w_in == w_out and h_in == h_out:
            return 0, 0, h_in, w_in

        top = random.randint(0, h_in - h_out)
        left = random.randint(0, w_in - w_out)
        return left, top, left + w_out, top + h_out

    def pad(self, img, pad_size, padding_mode="constant"):
        if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
            raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

        pad_left, pad_top, pad_right, pad_bottom = pad_size
        img = np.asarray(img)

        # RGB image
        if len(img.shape) == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
        # Grayscale image
        if len(img.shape) == 2:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)

        return Image.fromarray(img)


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
        'Flip': Flip,
        'Solarize': Solarize,
        'Posterize': Posterize,
        'Contrast': Contrast,
        'Color': Color,
        'Brightness': Brightness,
        'Sharpness': Sharpness,
        'RandomResizedCrop': RandomResizedCrop
    }
    return transform_registry[config.type](config)
