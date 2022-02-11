# Copyright (c) 2021, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import matplotlib.image as matimg
import numpy as np
import torch

import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser

FOREST_LABEL, NON_FOREST_LABEL, NULL_LABEL = 2, 1, 0


def main(config, args):
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    if config.resume:
        logger.info('Loading checkpoint: {} ...'.format(config.resume))
        resume_path = config.resume
    else:
        logger.info('Loading checkpoint: {} ...'.format(
            config['trainer']['pretrained_model']))
        resume_path = config['trainer']['pretrained_model']
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint, strict=False)

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # set districts and years
    data_path = config['inference_data_loader']['data_path']
    if hasattr(args, "districts"):
        all_districts = config.districts
    else:
        # get all districts in directory
        all_districts = set()
        files = os.listdir(data_path)
        for file in files:
            district = file.split('_')[-1][:-4]
            all_districts.add(district)
        all_districts = list(all_districts)
    if hasattr(args, "years"):
        years = config.years
    else:
        # get all years in directory
        years = set()
        files = os.listdir(data_path)
        for file in files:
            year = int(file.split('_')[1])
            years.add(year)
        years = list(years)

    for district in all_districts:
        for year in years:
            logger.info("On District: {} @ Year: {}".format(district, year))
            image_path = os.path.join(
                data_path, 'landsat8_{}_region_{}.tif'.format(year, district))
            # setup data_loader
            inference_loader = config.init_obj(
                'inference_data_loader', module_data, image_path, district)
            adjustment_mask = inference_loader.dataset.adjustment_mask
            # we need to fill our new generated test image
            generated_map = np.empty(
                shape=inference_loader.dataset.get_image_size())
            for idx, data in enumerate(inference_loader):
                coordinates, test_x = data['coordinates'].tolist(
                ), data['input']
                test_x = test_x.to(device)
                _, softmaxed = model.forward(test_x)
                pred = torch.argmax(softmaxed, dim=1)
                pred_numpy = pred.cpu().numpy().transpose(1, 2, 0)
                if idx % 5 == 0:
                    logger.info('On {} of {}'.format(
                        idx, len(inference_loader)))
                for k in range(test_x.shape[0]):
                    x, x_, y, y_ = coordinates[k]
                    generated_map[x:x_, y:y_] = pred_numpy[:, :, k]
            # adjust the inferred map
            generated_map += 1  # to make forest pixels: 2, non-forest pixels: 1, null pixels: 0
            generated_map = np.multiply(generated_map, adjustment_mask)
            # save generated map as png image, not numpy array
            forest_map_rband = np.zeros_like(generated_map)
            forest_map_gband = np.zeros_like(generated_map)
            forest_map_bband = np.zeros_like(generated_map)
            forest_map_gband[generated_map == FOREST_LABEL] = 255
            forest_map_rband[generated_map == NON_FOREST_LABEL] = 255
            forest_map_for_visualization = np.dstack(
                [forest_map_rband, forest_map_gband, forest_map_bband]).astype(np.uint8)
            save_this_map_path = os.path.join(
                config.inference_dir, '{}_{}_inferred_map.png'.format(district, year))
            matimg.imsave(save_this_map_path, forest_map_for_visualization)
            logger.info('Saved: {} @ {}'.format(save_this_map_path,
                        forest_map_for_visualization.shape))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Inference Script')
    args.add_argument('-c', '--config', default="./config.json", type=str,
                      help='config file path (default: ./config.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    args.add_argument('-dst', '--districts', nargs='+', type=str,
                      default=["abbottabad"], help='districts to consider')
    args.add_argument('-y', '--years', nargs='+', type=int,
                      default=[2016], help='years to consider')

    config = ConfigParser.from_args(args)
    main(config, args)
