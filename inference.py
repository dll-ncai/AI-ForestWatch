# Copyright (c) 2021, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


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
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()


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
