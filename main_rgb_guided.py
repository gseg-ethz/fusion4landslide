# ---------------------------------------------------------------------------- #
# Main functions of the RGB-guided method
# Refer to the ISPRS workshop paper for details.
# This version supports processing multiple tiles.
# ---------------------------------------------------------------------------- #
import argparse
from utils.common import load_yaml, dir_exist, access_device, run_time
from easydict import EasyDict as edict
from cpp_core.pcd_tiling.build import pcd_tiling
from cpp_core.supervoxel_segmentation.build import supervoxel
from src.functions import point_cloud_tiling
import numpy as np
import open3d as o3d
from utils.logger import get_logger
import yaml
from src.rgb_guided import Image_DVFs
import time
import torch
import glob
from tqdm import tqdm
import gc
import os
import os.path as osp
import re
import shutil
from utils.common import setup_seed
from easydict import EasyDict
import copy

import warnings
warnings.simplefilter("ignore", FutureWarning)  # Suppress FutureWarnings

setup_seed(0)  # Set random seed for reproducibility


def to_dict(obj):
    """
    Recursively converts an EasyDict object to a standard Python dictionary.
    """
    if isinstance(obj, EasyDict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
        return obj


def main():
    """
    Main function to execute the RGB-guided method.
    """
    # Load configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/test/demo.yaml',
                        help='Path to the configuration file.')
    args = parser.parse_args()
    cfg = load_yaml(args.config, keep_sub_directory=True)
    cfg = load_yaml(args.config, keep_sub_directory=True)

    # Ensure the output directory exists
    cfg['path_name']['output_root'] = osp.join(cfg['path_name']['output_dir'], cfg['path_name']['output_folder'])
    log_dir = osp.join(cfg['path_name']['output_root'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_save_path = osp.join(log_dir, f'rgb_guided_{time.strftime("%Y%m%d_%H%M%S")}.log')
    cfg['logging'] = get_logger(log_save_path)
    # dir_exist(log_dir, sub_folders=None)
    log_save_path = osp.join(log_dir, 'rgb_guided_{}.log'.format(time.strftime('%Y%m%d_%H%M%S')))
    cfg['logging'] = get_logger(log_save_path)

    cfg = edict(cfg)
    cfg.path_name.project_dir = osp.abspath(os.path.dirname(__file__))
    cfg.verbose = cfg.misc.verbose
    cfg.save_interim = cfg.misc.save_interim
    cfg.device = access_device()

    # Log configuration details
    cfg.logging.info('-' * 70)
    log_message = "Config: \n" + yaml.dump(load_yaml(args.config), sort_keys=False, default_flow_style=False, indent=2)
    cfg.logging.info(log_message)
    cfg.logging.info('-' * 70)

    with torch.no_grad():
        # recode the time spent
        start_time = time.time()

        # Perform point cloud tiling if required
        cfg.path_name.tile_dir = osp.join(cfg.path_name.output_root, 'tiled_data')
        dir_exist(cfg.path_name.tile_dir)
        if not any(os.listdir(cfg.path_name.tile_dir)):
            config = copy.deepcopy(cfg)
            config.data_dir = cfg.path_name.input_root
            config.src_name = cfg.data.src_pcd
            config.tgt_name = cfg.data.tgt_pcd
            config.tiling_type = cfg.method.tiling_type
            config.max_pts_per_tile = cfg.method.max_pts_per_tile
            config.min_pts_per_tile = cfg.method.min_pts_per_tile
            config.voxel_size = cfg.method.voxel_size
            config.tile_dir = cfg.path_name.tile_dir
            config.verbose = cfg.verbose
            point_cloud_tiling(config)
        else:
            cfg.logging.info('Skip point cloud tiling. Tiles will be loaded from %s.', cfg.path_name.tile_dir)

            # if cfg.method.tiling_type == 'xyz':
            # else:
            #     NotImplementedError

        # Load tiled files
        src_tiled_overlap_list = sorted(glob.glob(osp.join(cfg.path_name.tile_dir, 'overlap', "source_tile_*")),
                                        key=lambda x: int(re.search(r'\d+', osp.basename(x)).group()))
        if cfg.verbose:
            cfg.logging.info(f'Number of tiles from source/target point cloud: {len(src_tiled_overlap_list)}')

        # re_run
        # 40,
        # Process each tile
        continue_tile = 0
        for tile_i, src_tile_overlap_path in enumerate(tqdm(src_tiled_overlap_list[continue_tile:], position=0, leave=True)):
            cfg.logging.info(f'Processing tile {tile_i + continue_tile} of {len(src_tiled_overlap_list)}')
            tgt_tile_overlap_path = src_tile_overlap_path.replace('source_tile_', 'target_tile_')
            assert osp.exists(tgt_tile_overlap_path)
            tgt_tile_overlap_path = src_tile_overlap_path.replace('source_tile_', 'target_tile_')
            assert osp.exists(tgt_tile_overlap_path)

            cfg.tile_id = re.findall(r'\d+', osp.basename(src_tile_overlap_path))[0]
            cfg.src_tile_overlap_path = src_tile_overlap_path
            cfg.tgt_tile_overlap_path = tgt_tile_overlap_path

            # Start RGB-guided estimation
            pure_2d_matching = Image_DVFs(cfg)
            pure_2d_matching.implement_rgb_guided_estimation()

        end_time = time.time()

    if cfg.verbose:
            cfg.logging.info(f"Deformation analysis completed! Log saved to: '{log_save_path}'.")
            cfg.logging.info(f"Results saved to: '{cfg.path_name.output_root}'. "
                             f"Total time taken: {(end_time - start_time)/3600:.2f} hours "
                             f"or {(end_time - start_time):.1f} seconds.")


if __name__ == '__main__':
    main()
