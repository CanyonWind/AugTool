import random
from os import listdir
from os.path import isfile, isdir, join
from PIL import Image, ImageOps


class DataLoader:
    """
    A dataloader to retrieve images batch to batch.
    Args:
        source_dirs (list): List of source directories, ['./data/rgb', './data/depth', ...]
        img_names (list): List of image names.
        batch_size (int): Batch size.
        keep_last_batch (Bool): Whether to keep the last batch when its size < batch_size.
    """
    def __init__(self, source_dirs, img_names, batch_size=4, keep_last_batch=True):
        self.index = 0
        assert len(source_dirs) > 0, "Source directories should not be empty."
        self.source_dirs = source_dirs
        self.img_names = img_names
        self.batch_size = batch_size
        self.keep_last_batch = keep_last_batch

    def shuffle(self):
        random.shuffle(self.img_names)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        """
        Fetch next batch.
        Returns:
            list: List of image names for this batch
            list: List of list. The inner list contains the different sources, Images, for one instance.
                  The sequence of the inner list follows the given source_dirs.
        """
        if self.index >= len(self.img_names) or (not self.keep_last_batch and
                                                 len(self.img_names) - self.index < self.batch_size):
            raise StopIteration

        batch_size = min(len(self.img_names) - self.index, self.batch_size)
        batch = []
        img_names = []
        # load image batch and collate to numpy array
        for _ in range(batch_size):
            image_group = []
            for i, src_dir in enumerate(self.source_dirs):
                self.load(src_dir, self.img_names[self.index], image_group)
            batch.append(image_group)
            img_names.append(self.img_names[self.index])
            self.index += 1
        return img_names, batch

    @staticmethod
    def load(source_dir, img_name, target_pool):
        img = Image.open(join(source_dir, img_name))
        target_pool.append(img)
        return


if __name__ == '__main__':
    data_root = '../data'
    source_dirs = [join(data_root, dir_name) for dir_name in listdir(data_root)
                   if isdir(join(data_root, dir_name))]
    image_names = [img_name for img_name in listdir(join(data_root, 'rgb'))
                   if isfile(join(data_root, 'rgb', img_name))]
    dataloader = DataLoader(source_dirs, image_names)
    for batch in dataloader:
        img = batch[1][0][2]
        for _ in range(5):
            gen_rand_value = lambda: random.uniform(-45, 45)
            value = gen_rand_value()
            rotated = img.rotate(value)
            rotated = rotated.transform(
                rotated.size, Image.AFFINE, (1, 0.1, 0, 0, 1, 0))
            rotated = ImageOps.autocontrast(rotated)
            rotated.show()
