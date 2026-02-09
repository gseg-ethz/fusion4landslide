from PIL import Image
import os.path as osp
from utils.common import dir_exist
from easydict import EasyDict as edict
import argparse
from utils.common import setup_seed, load_yaml
import time

def crop_and_save(img, img_path, start_x, start_y, crop_size, output_path):
    # Crop the image
    img_cropped = img.crop((start_x, start_y, start_x + crop_size[0], start_y + crop_size[1]))

    # Save the cropped image
    save_dir = osp.join(output_path, 'cropped_images', osp.splitext(osp.basename(img_path))[0])
    dir_exist(save_dir)
    img_cropped.save(osp.join(save_dir, f'{start_x}_{start_y}.jpg'))
    img_cropped.save(osp.join(save_dir, f'{start_x}_{start_y}.jpg'))


def crop_image(config):
    config.src_img_path = osp.join(config.input_root, 'image', 'raw_images', config.src_image)
    config.tgt_img_path = osp.join(config.input_root, 'image', 'raw_images', config.tgt_image)

    src_img, tgt_img = Image.open(config.src_img_path), Image.open(config.tgt_img_path)
    # Calculate the starting coordinates for cropping
    start_x = 0
    while start_x + config.crop_size[0] <= src_img.width:
        start_y = 0
        while start_y + config.crop_size[1] <= src_img.height:
            crop_and_save(src_img, config.src_img_path, start_x, start_y, config.crop_size, config.output_root)
            crop_and_save(tgt_img, config.tgt_img_path, start_x, start_y, config.crop_size, config.output_root)

            # Move the cropping window
            start_y += config.crop_size[1] - config.overlap_size[1]
        start_x += config.crop_size[0] - config.overlap_size[0]

    print('Done! Cropped images saved in {}'.format(config.output_root))


def main():
    parser = argparse.ArgumentParser("Compute evaluation metrics")
    parser.add_argument('--config', type=str,
                        # default='../configs/preprocess/image_crop_rockfall_simulator.yaml',
                        default='../configs/preprocess/image_crop_brienz_tls.yaml',
                        help='Path to config file.')
    args = parser.parse_args()
    cfg = load_yaml(args.config, keep_sub_directory=False)
    cfg = edict(cfg)

    start_time = time.time()

    crop_image(cfg)

    end_time = time.time()

    print(f'Total time taken: {(end_time - start_time)/60:.2f} minutes or {(end_time - start_time):.1f} seconds')

    # define a dict
    # config = edict(dict())
    # Open the original image
    #############
    # for Brienz dataset, image_size = (1920, 2560)
    # config.input_dir = '/scratch2/zhawang/projects/deformation/DeformHD_local/data/Brienz/AOI_1/image/raw_images'
    # config.output_dir = '../output/Brienz/AOI_1'
    # # Define the crop size and overlap size
    # config.crop_size = (640, 1024)
    # config.overlap_size = (320, 512)
    #############
    # for Rockfall simulator, image_size = (5120, 5120)
    # config.input_dir = '/scratch2/zhawang/projects/deformation/DeformHD_local/data/Rockfall_Simulator/image/raw_images'
    # config.output_dir = '/scratch2/zhawang/projects/deformation/DeformHD_local/output/Rockfall_Simulator/pure_2d'
    #
    # config.src_img_path = osp.join(config.input_dir, 'epoch_1.jpg')
    # config.tgt_img_path = osp.join(config.input_dir, 'epoch_2.jpg')
    #
    # # Define the crop size and overlap size
    # # config.crop_size = (1024, 1024)
    # # config.overlap_size = (512, 512)
    #
    # config.crop_size = (1500, 1500)
    # config.overlap_size = (750, 1000)


if __name__ == '__main__':
    main()
