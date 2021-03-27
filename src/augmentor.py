from dataloader import DataLoader
from config import load_config
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

    def save_results(self, img_names, data, aug_idx):
        output_dir = self.config.output_dir

    def augment(self, img_names, data, aug_idx, save=False):
        """
        Take a batch of data and do augmentation sequentially according to the pipeline.
        Args:
            img_names (list): List of image names.
            data (dict): Dict of {src_name: src_batch} pair.
                         Each src_batch is a [N, H, W, C] numpy array.
            aug_idx (int): Augmentation index.
            save (bool): Whether to save the output after augmentation.
        """
        for trans_op in self.transform_pipeline:
            data = trans_op(data)
        if save:
            self.save_results(img_names, data, aug_idx)
        return data


if __name__ == '__main__':
    from os.path import isfile, isdir, join
    from os import listdir

    config = load_config('../config/synthetic_3d_config.py')
    data_root = config.data_root
    source_dirs = [join(data_root, dir_name) for dir_name in listdir(data_root)
                   if isdir(join(data_root, dir_name))]
    image_names = [img_name for img_name in listdir(join(data_root, 'rgb'))
                   if isfile(join(data_root, 'rgb', img_name))]

    # initialize dataloader and augmentor
    dataloader = DataLoader(source_dirs, image_names)
    augmentor = Augmentor(config)
    for batch_tuple in dataloader:
        img_names, batch = batch_tuple
        for aug_idx in range(config.aug_times):
            augmentor.augment(img_names, batch, aug_idx, save=True)
    print('U can really dance')

