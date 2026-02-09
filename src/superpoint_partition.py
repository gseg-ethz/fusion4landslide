# ---------------------------------------------------------------------------- #
# modified from official superpoint_transformer/notebooks/superpoint_transformer_tutorial.ipynb
# just run once, end-to-end to get the three levels of superpoint partition results
# input root, config file
# processing mattertal ROIs as an example
##
# output superpoint indices
# note that level0 is just voxelization of the input point cloud, not superpoint partition
# superpoint partition starts from level1 to level 3
# ideally, the voxel size should be near the average point cloud spacing
# ---------------------------------------------------------------------------- #

import hydra # pip install hydra-core
from superpoint_transformer.src.utils import init_config
from easydict import EasyDict as edict
from superpoint_transformer.src.transforms.data import NAGAddKeysTo
import os
import os.path as osp
import numpy as np
import torch
from colorhash import ColorHash
import open3d as o3d

# superpoint transformer/src/utils/color.py
def int_to_plotly_rgb(x):
    """Convert 1D torch.Tensor of int into plotly-friendly RGB format.
    This operation is deterministic on the int values.
    """
    assert isinstance(x, torch.Tensor)
    assert x.dim() == 1
    assert not x.is_floating_point()
    x = x.cpu().long().numpy()
    palette = np.array([ColorHash(i).rgb for i in range(x.max() + 1)])
    return palette[x]


def superpoint_partition(tile_id, partition_path, logging, device):
    # sematic, test, dales_11g, important for config hyper-parameters
    task_widget = 'semantic'
    experiment = 'dales_11g'
    # experiment = 's3dis'
    # not used in this script, no need for pretrained model
    pretrained_weight = ''

    #################
    # initialize configs
    # load corresponding 'yaml' from 'configs/experiment/semantic/dales_11g.yaml'
    cfg = init_config(overrides=[
        f"experiment={task_widget}/{experiment}",
        f"ckpt_path={pretrained_weight}",
        f"datamodule.load_full_res_idx={True}"  # only when you need full-resolution predictions
    ])

    # should write some changes of config parameters in the following
    # partition feature types, either xyz or xyzrgb
    # xyzrgb, xyz
    partition_type = 'xyzrgb'
    # partition high-level features
    if partition_type == 'xyzrgb':
        cfg.datamodule.partition_hf = ['linearity', 'planarity', 'scattering', 'intensity']
    elif partition_type == 'xyz':
        cfg.datamodule.partition_hf = ['linearity', 'planarity', 'scattering']
    else:
        raise ValueError(f'partition_type {partition_type} not recognized')

    config = edict(dict())
    config.data_dir = osp.dirname(partition_path)
    config.file_folder = 'tiled_data/overlap'
    config.tile_id = tile_id

    # Instantiate the datamodule
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data(config)
    datamodule.setup(config)

    # Pick among train, val, and test datasets. It is important to note that
    # the train dataset produces augmented spherical samples of large
    # scenes, while the val and test dataset load entire tiles at once
    dataset = datamodule.test_dataset

    # For the sake of visualization, we require that NAGAddKeysTo does not
    # remove input Data attributes after moving them to Data.x, so we may
    # visualize them
    for t in dataset.on_device_transform.transforms:
        if isinstance(t, NAGAddKeysTo):
            t.delete_after = False

    # Load the first dataset item. This will return the hierarchical
    # partition of an entire tile, as a NAG object
    for i in range(len(dataset)):
        nag = dataset[i]

        # Apply on-device transforms on the NAG object. For the train dataset,
        # this will select a spherical sample of the larger tile and apply some
        # data augmentations. For the validation and test datasets, this will
        # prepare an entire tile for inference
        nag = dataset.on_device_transform(nag.to(device))

        logging.info(f"Levels of ratios: '|P_0|/|P_1|' {nag.level_ratios['|P_0| / |P_1|']:.2f}, "
                     f"'|P_1|/|P_2|' {nag.level_ratios['|P_1| / |P_2|']:.2f}, "
                     f"'|P_2|/|P_3|' {nag.level_ratios['|P_2| / |P_3|']:.2f}")

        indices_raw_to_level0 = nag[0].sub.to_super_index()
        indices_level0_to_level1 = nag[0].super_index
        indices_level1_to_level2 = nag[1].super_index
        indices_level2_to_level3 = nag[2].super_index

        if i == 0:
            pts = o3d.io.read_point_cloud(
                osp.join(config.data_dir, config.file_folder, f'source_tile_{config.tile_id}_overlap.ply'))
        elif i == 1:
            pts = o3d.io.read_point_cloud(
                osp.join(config.data_dir, config.file_folder, f'target_tile_{config.tile_id}_overlap.ply'))
        else:
            raise ValueError('Only two tiles are available!')
        pts = np.asarray(pts.points)

        # colors_0 = int_to_plotly_rgb(indices_raw_to_level0)
        colors_1 = int_to_plotly_rgb(indices_level0_to_level1)
        colors_2 = int_to_plotly_rgb(indices_level1_to_level2)
        colors_3 = int_to_plotly_rgb(indices_level2_to_level3)

        indices_raw_to_level1 = torch.stack([indices_level0_to_level1[int(j)] for j in indices_raw_to_level0])
        indices_raw_to_level2 = torch.stack(
            [indices_level1_to_level2[int(indices_level0_to_level1[int(j)])] for j in indices_raw_to_level0])
        indices_raw_to_level3 = torch.stack(
            [indices_level2_to_level3[int(indices_level1_to_level2[int(indices_level0_to_level1[int(j)])])] for j in
             indices_raw_to_level0])

        # colors_raw_to_level0 = colors_0[indices_raw_to_level0.cpu(), :]
        colors_raw_to_level1 = colors_1[indices_raw_to_level1.cpu(), :]
        colors_raw_to_level2 = colors_2[indices_raw_to_level2.cpu(), :]
        colors_raw_to_level3 = colors_3[indices_raw_to_level3.cpu(), :]

        # optional, output individual layers
        output_path = osp.join(config.data_dir, 'superpoint_partition')
        os.makedirs(output_path, exist_ok=True)
        if i == 0:
            np.savetxt(osp.join(output_path, f'partition_of_input_src_tile_{tile_id}.txt'),
                       np.concatenate([pts, colors_raw_to_level1, indices_raw_to_level1.cpu()[:, None],
                                       colors_raw_to_level2, indices_raw_to_level2.cpu()[:, None],
                                       colors_raw_to_level3, indices_raw_to_level3.cpu()[:, None]], axis=1),
                       fmt='%.3f %.3f %.3f %d %d %d %d %d %d %d %d %d %d %d %d')
            # optional, output individual layers
            # # np.savetxt(osp.join(output_path, 'source_partition_level0.txt'),
            # #            np.concatenate([pts, colors_raw_to_level0, indices_raw_to_level0.cpu()[:, None]], axis=1),
            # #            fmt='%.3f %.3f %.3f %d %d %d %d')
            # np.savetxt(osp.join(output_path, 'source_partition_level1.txt'),
            #            np.concatenate([pts, colors_raw_to_level1, indices_raw_to_level1.cpu()[:, None]], axis=1),
            #            fmt='%.3f %.3f %.3f %d %d %d %d')
            # np.savetxt(osp.join(output_path, 'source_partition_level2.txt'),
            #            np.concatenate([pts, colors_raw_to_level2, indices_raw_to_level2.cpu()[:, None]], axis=1),
            #            fmt='%.3f %.3f %.3f %d %d %d %d')
            # np.savetxt(osp.join(output_path, 'source_partition_level3.txt'),
            #            np.concatenate([pts, colors_raw_to_level3, indices_raw_to_level3.cpu()[:, None]], axis=1),
            #            fmt='%.3f %.3f %.3f %d %d %d %d')
        elif i == 1:
            np.savetxt(osp.join(output_path, f'partition_of_input_tgt_tile_{tile_id}.txt'),
                       np.concatenate([pts, colors_raw_to_level1, indices_raw_to_level1.cpu()[:, None],
                                       colors_raw_to_level2, indices_raw_to_level2.cpu()[:, None],
                                       colors_raw_to_level3, indices_raw_to_level3.cpu()[:, None]], axis=1),
                       fmt='%.3f %.3f %.3f %d %d %d %d %d %d %d %d %d %d %d %d')
            # optional, output individual layers
            # # np.savetxt(osp.join(output_path, 'target_partition_level0.txt'),
            # #            np.concatenate([pts, colors_raw_to_level0, indices_raw_to_level0.cpu()[:, None]], axis=1),
            # #            fmt='%.3f %.3f %.3f %d %d %d %d')
            # np.savetxt(osp.join(output_path, 'target_partition_level1.txt'),
            #            np.concatenate([pts, colors_raw_to_level1, indices_raw_to_level1.cpu()[:, None]], axis=1),
            #            fmt='%.3f %.3f %.3f %d %d %d %d')
            # np.savetxt(osp.join(output_path, 'target_partition_level2.txt'),
            #            np.concatenate([pts, colors_raw_to_level2, indices_raw_to_level2.cpu()[:, None]], axis=1),
            #            fmt='%.3f %.3f %.3f %d %d %d %d')
            # np.savetxt(osp.join(output_path, 'target_partition_level3.txt'),
            #            np.concatenate([pts, colors_raw_to_level3, indices_raw_to_level3.cpu()[:, None]], axis=1),
            #            fmt='%.3f %.3f %.3f %d %d %d %d')
