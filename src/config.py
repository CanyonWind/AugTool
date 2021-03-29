import sys
import os.path as osp
import pprint
from importlib import import_module
from addict import Dict


def load_config(file_path, photo_metric_distortion_for_all):
    """
    Convert python config file to an addict.Dict.
    Args:
        file_path (str): config file path.
        photo_metric_distortion_for_all (bool): whether to apply photo metric distortion for all sources.
    Returns:
        addict.Dict: the configuration dict
    """
    temp_config_dir = file_path[:file_path.rfind('/')]
    temp_module_name = osp.splitext(file_path.split('/')[-1])[0]
    sys.path.insert(0, temp_config_dir)
    mod = import_module(temp_module_name)
    mod.photo_metric_distortion_for_all[0] = photo_metric_distortion_for_all
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    }
    # delete imported module
    del sys.modules[temp_module_name]
    return Dict(cfg_dict)


if __name__ == '__main__':
    config_path = '../configs/synthetic_3d_config.py'
    config = load_config(config_path)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print(config.aug_pipeline[0].type)
