# Copyright (c) 2021, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
from tqdm import tqdm
import matplotlib.image as matimg

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

FOREST_LABEL, NON_FOREST_LABEL, NULL_LABEL = 2, 1, 0


def main(config):
    logger = config.get_logger('test')

    # setup data_loader 
    # data_loader = config.init_obj('inference_data_loader', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    if config.resume:
        checkpoint = torch.load(config.resume)
    else:
        checkpoint = torch.load(config.trainer.pretrained_model)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict, strict=False)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    if config.districts:
        all_districts = config.districts
    else:
        # TODO
        # get all districts in directory
    if config.years:
        years = config.years
    else:
        # TODO
        # get all years in directory
        
    for district in all_districts:
        for year in years:
            print("(LOG): On District: {} @ Year: {}".format(district, year))
            image_path = os.path.join(args.data_path, 'landsat8_{}_region_{}.tif'.format(year, district)) #added(nauman)
            inference_loader = config.init_obj('inference_data_loader', module_data, image_path)
            adjustment_mask = inference_loader.dataset.adjustment_mask
            # we need to fill our new generated test image
            generated_map = np.empty(shape=inference_loader.dataset.get_image_size())
            for idx, data in enumerate(inference_loader):
                coordinates, test_x = data['coordinates'].tolist(), data['input']
                test_x = test_x.cuda(device=args.device) if args.cuda else test_x
                out_x, softmaxed = model.forward(test_x)
                pred = torch.argmax(softmaxed, dim=1)
                pred_numpy = pred.cpu().numpy().transpose(1,2,0)
                if idx % 5 == 0:
                    print('LOG: on {} of {}'.format(idx, len(inference_loader)))
                for k in range(test_x.shape[0]):
                    x, x_, y, y_ = coordinates[k]
                    generated_map[x:x_, y:y_] = pred_numpy[:,:,k]
            # adjust the inferred map
            generated_map += 1  # to make forest pixels: 2, non-forest pixels: 1, null pixels: 0
            generated_map = np.multiply(generated_map, adjustment_mask)
            # save generated map as png image, not numpy array
            forest_map_rband = np.zeros_like(generated_map)
            forest_map_gband = np.zeros_like(generated_map)
            forest_map_bband = np.zeros_like(generated_map)
            forest_map_gband[generated_map == FOREST_LABEL] = 255
            forest_map_rband[generated_map == NON_FOREST_LABEL] = 255
            forest_map_for_visualization = np.dstack([forest_map_rband, forest_map_gband, forest_map_bband]).astype(np.uint8)
            save_this_map_path = os.path.join(args.dest, '{}_{}_inferred_map.png'.format(district, year))
            matimg.imsave(save_this_map_path, forest_map_for_visualization)
            print('Saved: {} @ {}'.format(save_this_map_path, forest_map_for_visualization.shape))
    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Inference Script')
    args.add_argument('-d', '--districts', nargs='+', default=None, type=str,
                      default=["abbottabad"], help='districts to consider')
    args.add_argument('-y', '--years', nargs='+', default=None, type=int,
                      default=[2016], help='years to consider')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    config = ConfigParser.from_args(args)
    main(config)
