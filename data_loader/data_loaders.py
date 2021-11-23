# Copyright (c) 2021, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from torchvision import datasets, transforms
from base import BaseDataLoader, BaseDataset

random.seed(123)

class Landsat8DataLoader(BaseDataLoader):
    """
    Dataloader for Landsat8 generated pickle data
    """
    def __init__(self, data_dir, data_split_lists_path, batch_size, model_input_size, num_classes, one_hot, 
                 train_split=0.8, mode='train', num_workers=4, transforms=None):
        
        assert mode in ('train', 'val', 'test'), "Invalid value for train/val/test mode"
        
        # generated pickle data path
        self.data_dir = data_dir

        if not os.path.exists(data_split_lists_path):
            print('LOG: No saved data found. Making new data directory {}'.format(data_split_lists_path))
            os.mkdir(data_split_lists_path)
        
        full_examples_list = [os.path.join(extended_data_path, x) for x in os.listdir(extended_data_path)]
        random.shuffle(full_examples_list)
        train_split = int(train_split*len(full_examples_list))
               
        if mode == 'train':
            data_list = full_examples_list[:train_split]
        else:
            temp_list = full_examples_list[train_split:]
            if mode == 'val':
                data_list = temp_list[0:len(temp_list)//2]
            else:
                data_list = temp_list[len(temp_list)//2:]

        data_map_path = os.path.join(data_split_lists_path, f'{mode}_datamap.pkl')
        
        if mode == 'train':
            self.dataset = BaseDataset(data_list, data_map_path, 8, model_input_size, 
                                       bands, num_classes, one_hot, transforms=transforms)
            super().__init__(self.dataset, batch_size, True, num_workers)
        else:
            self.dataset = BaseDataset(data_list, data_map_path, model_input_size, model_input_size, 
                                       bands, num_classes, one_hot, mode='test', transforms=transforms)
            super().__init__(self.dataset, batch_size, False, num_workers)
        
