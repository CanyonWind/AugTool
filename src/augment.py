from os.path import isfile, isdir, join, splitext
from os import listdir, makedirs, mkdir
import argparse
from PIL import Image
from PIL import ImageMath

from config import load_config
from dataloader import DataLoader
from augmentor import Augmentor


def parse_args():
    parser = argparse.ArgumentParser(description='An augmentor to load, augment and save data from multi-sources.')
    parser.add_argument('--config', type=str,
                        default='./configs/synthetic_3d_config.py',
                        help='the config file path')
    parser.add_argument('--source-dirs', nargs='+',
                        help='<Required> Specify which sources to do augmentation', required=True)
    args = parser.parse_args()
    return args


def concat(im1, im2, im3):
    dst = Image.new('RGB', (im1.width + im2.width + im3.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(ImageMath.eval('im/256', {'im': im2}).convert('L'), (im1.width, 0))
    dst.paste(im3, (im1.width * 2, 0))
    return dst


def save_results(src_names, img_names, data, epoch, output_dir):
    makedirs(output_dir, exist_ok=True)
    for i, image_group in enumerate(data):
        img_name = img_names[i]
        for j, src_name in enumerate(src_names):
            save_image_dir = join(output_dir, '{}-{}'.format(src_name, splitext(img_name)[0]))
            img = image_group[j]
            if not isdir(save_image_dir):
                mkdir(save_image_dir)
            img.save(join(save_image_dir, '{:04d}.png'.format(epoch)))
        concat_img = concat(image_group[1], image_group[0], image_group[2])
        concat_dir = join(output_dir, 'concat-{}'.format(splitext(img_name)[0]))
        if not isdir(concat_dir):
            mkdir(concat_dir)
        concat_img.save(join(concat_dir, '{:04d}.png'.format(epoch)))
    return


def main():
    # TODO: add comment line inputs for
    #       source_dirs, config_path, data_root, output_dir, aug_times
    config = load_config(args.config)
    data_root = config.data.root
    source_dirs = args.source_dirs
    source_dirs = [join(data_root, dir_name) for dir_name in listdir(data_root)
                   if isdir(join(data_root, dir_name))]
    src_names = [src_dir.split('/')[-1] for src_dir in source_dirs]
    raw_input_idx = src_names.index('rgb') if 'rgb' in src_names else -1
    image_names = [img_name for img_name in listdir(source_dirs[0])
                   if isfile(join(source_dirs[0], img_name))]

    # initialize dataloader and augmentor
    dataloader = DataLoader(source_dirs, image_names)
    augmentor = Augmentor(config)
    for epoch in range(config.aug_times):
        if config.data.shuffle_load:
            dataloader.shuffle()
        print(f"Epoch: {epoch}")
        for i, batch_tuple in enumerate(dataloader):
            print(f"Augmenting batch {i}")
            img_names, batch = batch_tuple
            augmented = augmentor.augment(batch, raw_input_idx)
            save_results(src_names, img_names, augmented, epoch, config.data.output_dir)


if __name__ == '__main__':
    args = parse_args()
    main()
