from os.path import isfile, isdir, join, splitext
from os import listdir, makedirs, mkdir
from PIL import Image

from config import load_config
from dataloader import DataLoader
from augmentor import Augmentor


def concat(im1, im2, im3):
    dst = Image.new('RGB', (im1.width + im2.width + im3.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
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
        concat_img.save(join('../concat/', splitext(img_name)[0] + '.png'))
    return


def main():
    # TODO: add comment line inputs for
    #       source_dirs, config_path, data_root, output_dir, aug_times
    config = load_config('../configs/synthetic_3d_config.py')
    data_root = config.data.root
    source_dirs = [join(data_root, dir_name) for dir_name in listdir(data_root)
                   if isdir(join(data_root, dir_name))]
    src_names = [src_dir.split('/')[-1] for src_dir in source_dirs]
    raw_input_idx = src_names.index('rgb') if 'rgb' in src_names else -1
    image_names = [img_name for img_name in listdir(join(data_root, 'rgb'))
                   if isfile(join(data_root, 'rgb', img_name))]

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
    main()
