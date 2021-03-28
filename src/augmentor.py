import random
from transform import build_transform


class Augmentor:
    """
    Take batch of images and do augmentation over them.
    When config.pipeline == 'default', use the default pipeline for all operations. And the
    Augmentor.transform_pipeline is a list of transform.Transform.
    When config.pipeline == 'RL_searched', use the auto-aug searched pipeline for selected operations.
    And the Augmentor.transform_pipeline is a list of sub-policies, which are lists of transform.Transform.

    Args:
        config (addict.Dict): config specs.
    """
    def __init__(self, config):
        self.config = config
        self.transform_pipeline = self.build_pipeline()

    def build_pipeline(self):
        transform_pipeline = []
        if self.config.pipeline == 'default':
            for transform_config in self.config.default_pipeline:
                transform_pipeline.append(build_transform(transform_config))
        elif self.config.pipeline == 'RL_searched':
            for sub_policy in self.config.RL_searched_pipeline:
                sub_pipeline = []
                for transform_config in sub_policy:
                    sub_pipeline.append(build_transform(transform_config))
                transform_pipeline.append(sub_pipeline)
        else:
            raise ValueError(f"Expect pipeline types: (default, RL_searched), got {self.config.pipeline}")
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
            batch_pipeline = self.transform_pipeline if self.config.pipeline == 'default'\
                else random.choice(self.transform_pipeline)
            for trans_op in batch_pipeline:
                image_group = trans_op(image_group, raw_input_idx)
            augmented.append(image_group)
        return augmented


