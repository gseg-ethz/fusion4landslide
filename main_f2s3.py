import argparse
from utils.common import load_yaml, dir_exist, access_device, run_time
from easydict import EasyDict as edict
from cpp_core.pcd_tiling.build import pcd_tiling
from cpp_core.supervoxel_segmentation.build import supervoxel
from src.functions import point_cloud_tiling
# from src.median_res_compute import compute_median_resolution
import os.path as osp
import numpy as np
import open3d as o3d
from utils.logger import get_logger
from src.f2s3 import Deformation_Analyze
import time
import torch
import glob
from tqdm import tqdm
import gc
import os
import re
from src.models import PointNetFeature, FilteringNetwork

import warnings
warnings.simplefilter("ignore", FutureWarning)


def deformation_analysis_process(config):
    """
    The entire process of deformation analysis
    :return:
    """

    # point cloud tiling
    config.tile_dir = osp.join(config.output_dir, config.output_folder, 'tiled_data')
    if not osp.exists(config.tile_dir):
        point_cloud_tiling(config)
        if config.verbose:
            config.logging.info('Point cloud tiling is done. Save tiles to %s', config.tile_dir)
    else:
        if osp.exists(config.tile_dir):
            if config.verbose:
                config.logging.info('Skip point cloud tiling. Tiles will be loaded from: %s', config.tile_dir)
        else:
            raise NotImplementedError('Tile(s) do not exist in %s', osp.exists(config.tile_dir))


    # load tiled files
    src_tiled_overlap_list = sorted(glob.glob(osp.join(config.tile_dir, 'overlap', "source_tile_*")),
                                    key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
    if config.verbose:
        config.logging.info(f'Num. of tile(s) from source/target point cloud: {len(src_tiled_overlap_list)}', )

    config.device = access_device()
    load_pretrained_models(config)

    # implement a loop computation based each tile
    config.feat_dir = osp.join(config.output_dir, config.output_folder, 'features')
    if config.segment_type == 'supervoxel':
        config.segment_dir = osp.join(config.output_dir, config.output_folder, 'svl_segment')
    elif config.segment_type == 'superpoint':
        config.segment_dir = osp.join(config.output_dir, config.output_folder, 'spt_segment')
    config.corr_dir = osp.join(config.output_dir, config.output_folder, 'correspondences')
    dir_exist(config.feat_dir)
    dir_exist(config.segment_dir)
    dir_exist(config.corr_dir)
    continue_tile = 0
    for tile_i, src_tile_overlap_path in enumerate(tqdm(src_tiled_overlap_list[continue_tile:], ncols=150, desc='Load source tile overlaps')):
        tgt_tile_overlap_path = src_tile_overlap_path.replace('source_tile_', 'target_tile_')
        assert osp.exists(tgt_tile_overlap_path)

        config.tile_id = re.findall(r'\d+', os.path.basename(src_tile_overlap_path))[0]

        deformation_analysis = Deformation_Analyze(config, src_tile_overlap_path, tgt_tile_overlap_path)

        deformation_analysis.compute_features()

        deformation_analysis.implement_segmentation()

        deformation_analysis.correspondence_searching()

        if config.correspondence_pruning:
            deformation_analysis.correspondence_pruning()
        else:
            corres = deformation_analysis.correspondences
            magnitude = np.linalg.norm(corres[:, 3:6] - corres[:, :3], axis=1)
            output_path = osp.join(osp.dirname(config.corr_dir), 'results')
            np.savetxt(os.path.join(output_path, 'displacement_vector_and_magnitude_before_pruning.txt'),
                       np.hstack((corres[:, :3], magnitude[:, None])))

        # deformation_analysis.results_output()


def load_pretrained_models(config):
    # load the network of the feature descriptor
    if config.feat_compute and config.feat_type == 'DIPs':
        feat_model_path = osp.join(config.weight_dir, "local_feature_descriptor_best.pth")
        if config.verbose:
            config.logging.info('Load trained feature descriptor weight (%s) from: %s',
                              config.feat_type, feat_model_path)
        config.feat_desc_nn = PointNetFeature()
        config.feat_desc_nn.load_state_dict(torch.load(feat_model_path))
        config.feat_desc_nn.to(config.device)
        config.feat_desc_nn.eval()

    # load the network of the outlier removal
    if config.outlier_removal and config.removal_type == 'binary_classifier':
        outlier_removal_path = osp.join(config.weight_dir, 'outlier_classifier_best.pt')
        if config.verbose:
            config.logging.info('Load trained outlier removal weight (%s) from : %s',
                                config.removal_type, outlier_removal_path)
        config.outlier_removal_nn = FilteringNetwork()
        config.outlier_removal_nn.load_state_dict(torch.load(outlier_removal_path))
        config.outlier_removal_nn.to(config.device)
        config.outlier_removal_nn.eval()
    return None


def main():
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        # default='./configs/default_config.yaml',
                        # Mattertal
                        # default='./configs/landslide/f2s3_mattertal.yaml',
                        # Brienz
                        default='./configs/landslide/f2s3_brienz.yaml',
                        # Change detection
                        # default='./configs/change_detection/f2s3_fbk_landslide.yaml',
                        help='Path to config file.')
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    # cfg['snapshot_dir'] = 'snapshot/%s' % cfg['exp_dir']

    # ensure the log dir exists, otherwise create a dir
    log_dir = osp.join(cfg['output_dir'], cfg['output_folder'], 'logs')
    dir_exist(log_dir)
    log_save_path = osp.join(log_dir, 'deform_analysis_{}.log'.format(time.strftime('%Y%m%d_%H%M%S')))
    cfg['logging'] = get_logger(log_save_path)

    cfg = edict(cfg)

    # print log info to the terminal
    cfg.logging.info('----------------------------------------------------------------------')
    log_message = "Config: \n"
    for key, value in cfg.items():
        log_message += f"{key}={value} \n"
    cfg.logging.info(log_message)
    cfg.logging.info('----------------------------------------------------------------------')

    with torch.no_grad():
        # recode the time spent
        start_time = time.time()

        # start deformation analysis
        deformation_analysis_process(cfg)

        end_time = time.time()

    if cfg.verbose:
        cfg.logging.info(f"Deformation analysis is done! Save log information to: '{log_save_path}'.")
        cfg.logging.info(f"Save results to: '{cfg.output_dir}/{cfg.output_folder}'. Total time taken: {(end_time - start_time)/3600:.2f}"
                         f" hours or {(end_time - start_time):.1f} seconds.")


if __name__ == '__main__':
    main()
