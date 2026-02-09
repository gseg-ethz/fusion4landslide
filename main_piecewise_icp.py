import argparse
from utils.common import load_yaml, dir_exist, access_device, run_time
from easydict import EasyDict as edict
from src.functions import point_cloud_tiling
import os.path as osp
from utils.logger import get_logger
import time
import glob
from tqdm import tqdm
import os
import re
from utils.common import setup_seed
import copy
from src.piecewise_icp import Piecewise_ICP


setup_seed(0)


def main():
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        # default='./configs/default_config.yaml',
                        # Brienz
                        default='./configs/landslide/piecewise_icp_brienz.yaml',
                        # Mattertal
                        # default='./configs/landslide/piecewise_icp_mattertal.yaml',

                        help='Path to config file.')
    args = parser.parse_args()
    cfg = load_yaml(args.config, keep_sub_directory=False)

    # ensure the log dir exists, otherwise create a dir
    cfg['output_root'] = osp.join(cfg['output_dir'], cfg['output_folder'])

    # cfg['path_name']['output_root2'] = osp.join(cfg['path_name']['output_dir'], cfg['path_name']['output_folder'])
    log_dir = osp.join(cfg['output_root'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    # dir_exist(log_dir, sub_folders=None)
    log_save_path = osp.join(log_dir, 'piecewise_icp_{}.log'.format(time.strftime('%Y%m%d_%H%M%S')))
    cfg['logging'] = get_logger(log_save_path)

    cfg = edict(cfg)
    cfg.project_dir = osp.abspath(os.path.dirname(__file__))
    cfg.verbose = cfg.verbose
    cfg.save_interim = cfg.save_interim
    cfg.device = access_device()

    cfg.logging.info('-' * 70)
    log_message = "Config: \n"
    for key, value in cfg.items():
        log_message += f"{key}={value} \n"
    cfg.logging.info(log_message)
    cfg.logging.info('-' * 70)

    # recode the time spent
    start_time = time.time()

    cfg.tile_dir = osp.join(cfg.output_root, 'tiled_data')
    dir_exist(cfg.tile_dir)
    if not any(os.listdir(cfg.tile_dir)):
        config = copy.deepcopy(cfg)
        config.data_dir = cfg.input_root
        config.src_name = cfg.src_pcd
        config.tgt_name = cfg.tgt_pcd
        config.tiling_type = cfg.tiling_type
        config.max_pts_per_tile = cfg.max_pts_per_tile
        config.min_pts_per_tile = cfg.min_pts_per_tile
        config.voxel_size = cfg.voxel_size
        config.tile_dir = cfg.tile_dir
        config.verbose = cfg.verbose
        point_cloud_tiling(config)
    else:
        cfg.logging.info('Skip point cloud tiling. Tiles will be loaded from %s.', cfg.tile_dir)

    # load tiled files
    src_tiled_overlap_list = sorted(glob.glob(osp.join(cfg.tile_dir, 'overlap', "source_tile_*")),
                                    key=lambda x: int(re.search(r'\d+', osp.basename(x)).group()))
    if cfg.verbose:
        cfg.logging.info(f'Num. of tile(s) from source/target point cloud: {len(src_tiled_overlap_list)}', )

    # re_run
    continue_tile = 0
    for tile_i, src_tile_overlap_path in enumerate(tqdm(src_tiled_overlap_list[continue_tile:], position=0, leave=True)):
        cfg.logging.info(f'Current tile {tile_i + continue_tile} of total {len(src_tiled_overlap_list[:])} tiles')
        tgt_tile_overlap_path = src_tile_overlap_path.replace('source_tile_', 'target_tile_')
        assert osp.exists(tgt_tile_overlap_path)

        cfg.tile_id = re.findall(r'\d+', osp.basename(src_tile_overlap_path))[0]
        cfg.src_tile_overlap_path = src_tile_overlap_path
        cfg.tgt_tile_overlap_path = tgt_tile_overlap_path

        Piecewise_ICP(cfg)

    end_time = time.time()

    if cfg.verbose:
        cfg.logging.info(f"Displacement estimation is done! Save log information to: '{log_save_path}'.")
        cfg.logging.info(f"Save results to: '{cfg.output_root}'. "
                         f"Total time taken: {(end_time - start_time)/3600:.2f}"
                         f" hours or {(end_time - start_time):.1f} seconds.")


if __name__ == '__main__':
    main()
