# Copyright (c) 2021, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

np.random.seed(123)

class BaseDataset(Dataset):
    def __init__(self, data_list, data_map_path, stride, model_input_size, bands, num_classes, one_hot,
                 mode='train', transformation=None):
        super(dataset, self).__init__()
        self.data_list = data_list
        self.stride = stride
        self.model_input_size = model_input_size
        self.bands = [x-1 for x in bands]
        self.num_classes = num_classes
        self.one_hot = one_hot
        
        self.transformation = transformation
        self.mode = mode
        
        self.all_images = []
        self.total_images = 0

        if os.path.exists(data_map_path):
            print('LOG: Saved data map found! Loading now...')
            with open(data_map_path, 'rb') as data_map:
                self.data_list, self.all_images = pickle.load(data_map, encoding='latin1')
                self.total_images = len(self.all_images)
        else:
            print('LOG: No data map found! Generating now...')
            for example_path in self.data_list:
                with open(example_path, 'rb') as this_data:
                    _, label = pickle.load(this_data, encoding='latin1')
                    # we skip a label with too few valid pixels (<0.01*128*128)
                    if np.count_nonzero(label) > 160:
                      row_limit, col_limit = label.shape[0]-model_input_size, label.shape[1]-model_input_size
                      del label, _  # clear memory
                      for i in range(0, row_limit, self.stride):
                          for j in range(0, col_limit, self.stride):
                              self.all_images.append((example_path, i, j))
                              self.total_images += 1
            with open(data_map_path, 'wb') as data_map:
                pickle.dump((self.data_list, self.all_images), file=data_map)  # , protocol=pickle.HIGHEST_PROTOCOL)
                print('LOG: {} saved!'.format(data_map_path))

    def __getitem__(self, k):
        k = k % self.total_images
        (example_path, this_row, this_col) = self.all_images[k]

        with open(example_path, 'rb') as this_pickle:
            (example_subset, label_subset) = pickle.load(this_pickle, encoding='latin1')
            example_subset = np.nan_to_num(example_subset)
            label_subset = np.nan_to_num(label_subset)
        this_example_subset = example_subset[this_row:this_row + self.model_input_size, this_col:this_col + self.model_input_size, :]
        # get more indices to add to the example, landsat-8
        this_example_subset = get_indices(this_example_subset)
        # at this point, we pick which bands to use
        this_example_subset = this_example_subset[:, :, self.bands]
        this_label_subset = label_subset[this_row:this_row + self.model_input_size, this_col:this_col + self.model_input_size]
        if self.mode == 'train':
            # Convert NULL-pixels to Non-Forest Class only during training
            this_label_subset = fix(this_label_subset).astype(np.uint8)
            # augmentation
            if np.random.randint(0, 2) == 0:
                this_example_subset = np.fliplr(this_example_subset).copy()
                this_label_subset = np.fliplr(this_label_subset).copy()
            if np.random.randint(0, 2) == 1:
                this_example_subset = np.flipud(this_example_subset).copy()
                this_label_subset = np.flipud(this_label_subset).copy()
            if np.random.randint(0, 2) == 1:
                this_example_subset = np.fliplr(this_example_subset).copy()
                this_label_subset = np.fliplr(this_label_subset).copy()
            if np.random.randint(0, 2) == 0:
                this_example_subset = np.flipud(this_example_subset).copy()
                this_label_subset = np.flipud(this_label_subset).copy()
        if self.one_hot:
            this_label_subset = np.eye(self.num_classes)[this_label_subset.astype(int)]
        this_example_subset, this_label_subset = toTensor(image=this_example_subset, label=this_label_subset, one_hot=self.one_hot)
        if self.transformation:
            this_example_subset = self.transformation(this_example_subset)
        return {'input': this_example_subset, 'label': this_label_subset, 'sample_identifier': (example_path, this_row, this_col)}

    def __len__(self):
        return 1*self.total_images if self.mode == 'train' else self.total_images

def get_indices(arr):
    ndvi_band = (arr[:, :, 4] - arr[:, :, 3]) / (arr[:, :, 4] + arr[:, :, 3] + 1e-7)
    evi_band = 2.5 * (arr[:, :, 4] - arr[:, :, 3]) / (arr[:, :, 4] + 6 * arr[:, :, 3] - 7.5 * arr[:, :, 1] + 1)
    savi_band = 1.5 * (arr[:, :, 4] - arr[:, :, 3]) / (arr[:, :, 4] + v[:, :, 3] + 0.5)
    msavi_band = 0.5 * (2 * arr[:, :, 4] + 1 - np.sqrt((2 * arr[:, :, 4] + 1) ** 2 - 8 * (arr[:, :, 4] - arr[:, :, 3])))
    ndmi_band = (arr[:, :, 4] - arr[:, :, 5]) / (arr[:, :, 4] + arr[:, :, 5] + 1e-7)
    nbr_band = (arr[:, :, 4] - arr[:, :, 6]) / (arr[:, :, 4] + arr[:, :, 6] + 1e-7)
    nbr2_band = (arr[:, :, 5] - arr[:, :, 6]) / (arr[:, :, 5] + arr[:, :, 6] + 1e-7)
    arr = np.dstack((arr, np.nan_to_num(ndvi_band)))
    arr = np.dstack((arr, np.nan_to_num(evi_band)))
    arr = np.dstack((arr, np.nan_to_num(savi_band)))
    arr = np.dstack((arr, np.nan_to_num(msavi_band)))
    arr = np.dstack((arr, np.nan_to_num(ndmi_band)))
    arr = np.dstack((arr, np.nan_to_num(nbr_band)))
    arr = np.dstack((arr, np.nan_to_num(nbr2_band)))
    return arr

def fix(target_image):
    # we fix the label by
    # 1. Converting all NULL (0) pixels to Non-forest pixels (1)
    target_image[target_image == 0] = 1  # this will convert all null pixels to non-forest pixels
    # 2. Subtracting 1 from all labels => Non-forest = 0, Forest = 1
    target_image -= 1
    return target_image


def toTensor(**kwargs):
    image, label = kwargs['image'], kwargs['label']
    'will convert image and label from numpy to torch tensor'
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    if kwargs['one_hot']:
        label = label.transpose((2, 0, 1))
        return torch.from_numpy(image).float(), torch.from_numpy(label).float()
    return torch.from_numpy(image).float(), torch.from_numpy(label).long()
