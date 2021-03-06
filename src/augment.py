from os.path import isfile, isdir, join, splitext
from os import listdir, makedirs, mkdir
import argparse
from PIL import Image, ImageMath

from config import load_config
from dataloader import DataLoader
from augmentor import Augmentor


def parse_args():
    parser = argparse.ArgumentParser(description='An augmentor to load, augment and save data from multi-sources.')
    parser.add_argument('--config', type=str,
                        default='./configs/synthetic_3d_config.py',
                        help='The config file path.')
    parser.add_argument('--source-dirs', nargs='+',
                        help='<Required> Specify which sources to do augmentation', required=True)
    parser.add_argument('--output-dir', type=str, help='The output directory path.')
    parser.add_argument('--count', type=int, help='Number of times to do augmentation.')
    parser.add_argument('--photo-distort-all', action='store_true', help='Whether to do photo metric distortion'
                        'for all sources. If not, only RGB will be applied with photo metric distortions.')
    parser.add_argument('--shuffle-load', action='store_true', help='Whether to shuffle the data before loading batches.')
    parser.add_argument('--pipeline', type=str, choices=['default', 'RL_searched'], help='Which pipeline to apply.')
    parser.add_argument('--batch-size', type=int, help='Batch size.')

    args = parser.parse_args()
    return args


def concat(img1, img2, img3):
    dst = Image.new('RGB', (img1.width + img2.width + img3.width, img1.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.width, 0))
    dst.paste(img3, (img1.width * 2, 0))
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
            img.save(join(save_image_dir, '{:04d}.png'.format(epoch)), compress_level=1)
    return


def main():
    config = load_config(args.config, args.photo_distort_all)
    if args.output_dir:
        config.data.output_dir = args.output_dir
    if args.count:
        config.aug_times = args.count
    if args.shuffle_load:
        config.data.shuffle = args.shuffle_load
    if args.batch_size:
        config.data.batch_size = args.batch_size

    source_dirs = args.source_dirs
    src_names = [src_dir.split('/')[-1] for src_dir in source_dirs]
    raw_input_idx = src_names.index('rgb') if 'rgb' in src_names else -1
    image_names = [img_name for img_name in listdir(source_dirs[0])
                   if isfile(join(source_dirs[0], img_name))]

    # initialize dataloader and augmentor
    dataloader = DataLoader(source_dirs, image_names, batch_size=config.data.batch_size)
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
