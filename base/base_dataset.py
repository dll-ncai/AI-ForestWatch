# Copyright (c) 2021, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
import gdal
import torch
from torch.utils.data import Dataset

from datagen import mask_landsat8_image_using_rasterized_shapefile

np.random.seed(123)

class BaseTrainDataset(Dataset):
    def __init__(self, data_list, data_map_path, stride, model_input_size, bands, num_classes, one_hot,
                 mode='train', transforms=None):
        super(BaseDataset, self).__init__()
        self.data_list = data_list
        self.stride = stride
        self.model_input_size = model_input_size
        self.bands = [x-1 for x in bands]
        self.num_classes = num_classes
        self.one_hot = one_hot
        
        self.transforms = transforms
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
                      row_limit, col_limit = label.shape[0]-model_input_size, label.shape[1] - model_input_size
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
        if self.transforms:
            this_example_subset = self.transforms(this_example_subset)
        return {'input': this_example_subset, 'label': this_label_subset, 'sample_identifier': (example_path, this_row, this_col)}

    def __len__(self):
        return 1 * self.total_images if self.mode == 'train' else self.total_images
    
class BaseInferenceDataset(Dataset):
    def __init__(self, rasterized_shapefiles_path, image_path, bands, model_input_size, district, num_classes, transformation):
        super(dataset, self).__init__()
        self.model_input_size = model_input_size
        self.image_path = image_path
        self.all_images = []
        self.total_images = 0
        self.stride = stride
        self.bands = [int(this_band) - 1 for this_band in bands]  # 1-18 -> 0-17
        self.num_classes = num_classes
        self.transformation = transformation
        self.temp_dir = 'temp_numpy_saves'
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.mkdir(self.temp_dir)
        print('LOG: Generating data map now...')
        image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
        all_raster_bands = [image_ds.GetRasterBand(x+1).ReadAsArray() for x in range(image_ds.RasterCount)]


        # mask the image and adjust its size at this point
        test_image, self.adjustment_mask = mask_landsat8_image_using_rasterized_shapefile(rasterized_shapefiles_path=rasterized_shapefiles_path,
                                                                                          district=district, this_landsat8_bands_list=all_raster_bands)
        temp_image_path = os.path.join(self.temp_dir, 'temp_image.npy')
        np.save(temp_image_path, test_image)
        self.temp_test_image = np.load(temp_image_path, mmap_mode='r')
        row_limit = self.temp_test_image.shape[0] - model_input_size
        col_limit = self.temp_test_image.shape[1] - model_input_size
        test_image, image_ds, all_raster_bands = [None] * 3  # release memory
        for i in range(0, row_limit+1, self.stride):
            for j in range(0, col_limit+1, self.stride):
                self.all_images.append((i, j))
                self.total_images += 1
        self.shape = [i+self.stride, j+self.stride]

    def __getitem__(self, k):
        (this_row, this_col) = self.all_images[k]
        this_example_subset = self.temp_test_image[this_row:this_row + self.model_input_size, this_col:this_col + self.model_input_size, :]
        # get more indices to add to the example, landsat-8
        this_example_subset = get_indices(this_example_subset)
        # at this point, we pick which bands to forward based on command-line argument
        this_example_subset = this_example_subset[:, :, self.bands]
        this_example_subset = toTensor(image=this_example_subset)
        x1, x2 = this_row, this_row + self.model_input_size
        y1, y2 = this_col, this_col + self.model_input_size
        return {'coordinates': np.asarray([x1, x2, y1, y2]),
                'input': this_example_subset}

    def __len__(self):
        return self.total_images

    def get_image_size(self):
        return self.shape

    def clear_mem(self):
        shutil.rmtree(self.temp_dir)
        print('Log: Temporary memory cleared')

def get_indices(arr):
    bands = {
        "ndvi": (arr[:, :, 4] - arr[:, :, 3]) / (arr[:, :, 4] + arr[:, :, 3] + 1e-7),
        "evi": 2.5 * (arr[:, :, 4] - arr[:, :, 3]) / (arr[:, :, 4] + 6 * arr[:, :, 3] - 7.5 * arr[:, :, 1] + 1),
        "savi": 1.5 * (arr[:, :, 4] - arr[:, :, 3]) / (arr[:, :, 4] + arr[:, :, 3] + 0.5),
        "msavi": 0.5 * (2 * arr[:, :, 4] + 1 - np.sqrt((2 * arr[:, :, 4] + 1) ** 2 - 8 * (arr[:, :, 4] - arr[:, :, 3]))),
        "ndmi": (arr[:, :, 4] - arr[:, :, 5]) / (arr[:, :, 4] + arr[:, :, 5] + 1e-7),
        "nbr": (arr[:, :, 4] - arr[:, :, 6]) / (arr[:, :, 4] + arr[:, :, 6] + 1e-7),
        "nbr2": (arr[:, :, 5] - arr[:, :, 6]) / (arr[:, :, 5] + arr[:, :, 6] + 1e-7),
    }
    for name in bands:
        value = np.nan_to_num(bands[name])
        arr = np.dstack((arr, value))
    return arr

def fix(target_image):
    # we fix the label by
    # 1. Converting all NULL (0) pixels to Non-forest pixels (1)
    target_image[target_image == 0] = 1  # this will convert all null pixels to non-forest pixels
    # 2. Subtracting 1 from all labels => Non-forest = 0, Forest = 1
    target_image -= 1
    return target_image


def toTensor(image, label, one_hot=True):
    '''will convert image and label from numpy to torch tensor'''
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(image).float()
    if one_hot:
        label = label.transpose((2, 0, 1))
        label_tensor = torch.from_numpy(label).float()
    else:
        label_tensor = torch.from_numpy(label).long()
    return img_tensor, label_tensor
