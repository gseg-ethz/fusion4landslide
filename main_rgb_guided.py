# ---------------------------------------------------------------------------- #
# Main functions of the rgb-guided method, refer to the ISPRS workshop paper
# This is an updated version of the one mentioned in the workshop paper, supporting processing multiple tiles
# clarify the differences between wo_pruning, wo_refinement, and w_refinement output
# ---------------------------------------------------------------------------- #
import argparse
from utils.common import load_yaml, dir_exist, access_device, run_time
from easydict import EasyDict as edict
from src.functions import point_cloud_tiling
import os.path as osp
from utils.logger import get_logger
import yaml
from src.rgb_guided import Image_DVFs
import time
import torch
import glob
from tqdm import tqdm
import os
import re
from utils.common import setup_seed
from easydict import EasyDict
import copy

import warnings
warnings.simplefilter("ignore", FutureWarning)

setup_seed(0)


def to_dict(obj):
    if isinstance(obj, EasyDict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    else:
        return obj


def main():
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        # default='./configs/default_config.yaml',
                        # Brienz
                        default='./configs/landslide/rgb_guided_brienz.yaml',
                        # Mattertal
                        # default='./configs/landslide/rgb_guided_mattertal.yaml',
                        help='Path to config file.')
    args = parser.parse_args()
    cfg = load_yaml(args.config, keep_sub_directory=True)

    # ensure the log dir exists, otherwise create a dir
    cfg['path_name']['output_root'] = osp.join(cfg['path_name']['output_dir'], cfg['path_name']['output_folder'])

    log_dir = osp.join(cfg['path_name']['output_root'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_save_path = osp.join(log_dir, 'rgb_guided_{}.log'.format(time.strftime('%Y%m%d_%H%M%S')))
    cfg['logging'] = get_logger(log_save_path)

    cfg = edict(cfg)
    cfg.path_name.project_dir = osp.abspath(os.path.dirname(__file__))
    cfg.verbose = cfg.misc.verbose
    cfg.save_interim = cfg.misc.save_interim
    cfg.device = access_device()

    cfg.logging.info('-' * 70)
    log_message = "Config: \n" + yaml.dump(load_yaml(args.config), sort_keys=False, default_flow_style=False, indent=2)
    cfg.logging.info(log_message)
    cfg.logging.info('-' * 70)

    with torch.no_grad():
        # recode the time spent
        start_time = time.time()

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

        # load tiled files
        src_tiled_overlap_list = sorted(glob.glob(osp.join(cfg.path_name.tile_dir, 'overlap', "source_tile_*")),
                                        key=lambda x: int(re.search(r'\d+', osp.basename(x)).group()))
        if cfg.verbose:
            cfg.logging.info(f'Num. of tile(s) from source/target point cloud: {len(src_tiled_overlap_list)}', )

        continue_tile = 0
        for tile_i, src_tile_overlap_path in enumerate(tqdm(src_tiled_overlap_list[continue_tile:], position=0, leave=True)):

            cfg.logging.info(f'Current tile {tile_i + continue_tile} of total {len(src_tiled_overlap_list[:])} tiles')
            tgt_tile_overlap_path = src_tile_overlap_path.replace('source_tile_', 'target_tile_')
            assert osp.exists(tgt_tile_overlap_path)

            cfg.tile_id = re.findall(r'\d+', osp.basename(src_tile_overlap_path))[0]
            cfg.src_tile_overlap_path = src_tile_overlap_path
            cfg.tgt_tile_overlap_path = tgt_tile_overlap_path

            # start rgb_guided estimation
            pure_2d_matching = Image_DVFs(cfg)
            pure_2d_matching.implement_rgb_guided_estimation()

        end_time = time.time()

    if cfg.verbose:
        cfg.logging.info(f"Displacement estimation is done! Save log information to: '{log_save_path}'.")
        cfg.logging.info(f"Save results to: '{cfg.path_name.output_root}'. "
                         f"Total time taken: {(end_time - start_time)/3600:.2f}"
                         f" hours or {(end_time - start_time):.1f} seconds.")


if __name__ == '__main__':
    main()
