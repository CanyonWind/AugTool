from collections import defaultdict
from transform import build_transform


class Augmentor:
    """
    Take batch of images and do augmentation over them.
    Args:
        config (addict.Dict): config specs.
    """
    def __init__(self, config):
        self.config = config
        self.transform_pipeline = self.build_pipeline()

    def build_pipeline(self):
        transform_pipeline = []
        for transform_config in self.config.aug_pipeline:
            transform_pipeline.append(build_transform(transform_config))
        return transform_pipeline

    def augment(self, data, raw_input_idx):
        """
        Take a batch of data and do augmentation sequentially according to the pipeline.
        Args:
            data (list): List of list. The inner list contains the different sources, Images, for one instance.
                  The sequence of the inner list follows the given source_dirs.
            raw_input_idx (int): The index of the rgb input. -1 means rgb not existed.
        """
        augmented = []
        cur_batch_size = len(data)
        for i in range(cur_batch_size):
            image_group = data[i]
            for trans_op in self.transform_pipeline:
                image_group = trans_op(image_group, raw_input_idx)
            augmented.append(image_group)
        return augmented


