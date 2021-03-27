from collections import defaultdict
import cv2
import numpy as np


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

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        """
        Fetch next batch.
        Returns:
            list: List of image names for this batch
            dict: Dict of {src_name: src_batch} pair. Each src_batch is a [N, H, W, C] numpy array.
        """
        if self.index >= len(self.img_names) or (not self.keep_last_batch and
                                                 len(self.img_names) - self.index < self.batch_size):
            raise StopIteration

        batch_size = min(len(self.img_names) - self.index, self.batch_size)
        batch = defaultdict(list)
        dir2name = {src_dir: src_dir.split('/')[-1] for src_dir in self.source_dirs}
        img_names = []
        # load image batch and collate to numpy array
        for _ in range(batch_size):
            for src_dir in self.source_dirs:
                src = dir2name[src_dir]
                self.load(src_dir, self.img_names[self.index], batch[src])
            img_names.append(self.img_names[self.index])
            self.index += 1
        for src_dir in self.source_dirs:
            src = dir2name[src_dir]
            batch[src] = np.stack(batch[src])

        return img_names, batch

    @staticmethod
    def load(source_dir, img_name, target_pool):
        img = cv2.imread(join(source_dir, img_name))
        target_pool.append(img)
        return


if __name__ == '__main__':
    from os.path import isfile, isdir, join
    from os import listdir
    data_root = '../data'
    source_dirs = [join(data_root, dir_name) for dir_name in listdir(data_root)
                   if isdir(join(data_root, dir_name))]
    image_names = [img_name for img_name in listdir(join(data_root, 'rgb'))
                   if isfile(join(data_root, 'rgb', img_name))]
    dataloader = DataLoader(source_dirs, image_names)
    for batch in dataloader:
        print(batch)
    print('U can really dance')
