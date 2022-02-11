# Copyright (c) 2021, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from PIL import Image


def adaptive_resize(array, new_shape):
    # reshape the labels to the size of the image
    single_band = Image.fromarray(array)
    single_band_resized = single_band.resize(new_shape, Image.NEAREST)
    return np.asarray(single_band_resized)


def fix(target_image):
    # we fix the label by
    # 1. Converting all NULL (0) pixels to Non-forest pixels (1)
    # this will convert all null pixels to non-forest pixels
    target_image[target_image == 0] = 1
    # 2. Subtracting 1 from all labels => Non-forest = 0, Forest = 1
    target_image -= 1
    return target_image


def get_images_from_large_file(data_directory_path, label_directory_path, destination,
                               bands, year, region, stride):
    image_path = os.path.join(
        data_directory_path, 'landsat8_{}_region_{}.tif'.format(year, region))
    label_path = os.path.join(label_directory_path,
                              '{}_{}.tif'.format(region, year))
    if not os.path.exists(destination):
        print('Log: Making parent directory: {}'.format(destination))
        os.mkdir(destination)
    print(image_path, label_path)
    # we will use this to divide those fnf images
    covermap = gdal.Open(label_path, gdal.GA_ReadOnly)
    channel = covermap.GetRasterBand(1)
    label = channel.ReadAsArray()
    image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    x_size, y_size = image_ds.RasterXSize, image_ds.RasterYSize
    # we need the difference of the two raster sizes to do the resizing
    label = adaptive_resize(label, new_shape=(x_size, y_size))
    all_raster_bands = [image_ds.GetRasterBand(x) for x in bands]
    count = 1
    for i in range(y_size//stride):
        for j in range(x_size//stride):
            # read the label and drop this sample if it has all null pixels
            label_subset = label[i*stride:(i+1)*stride, j*stride:(j+1)*stride]
            # 0.01*256*256 ~ 650 pixels i.e at least 1% pixels should be valid
            if np.count_nonzero(label_subset) < 600:
                print("(LOG): Dropping NULL Pixel Sample")
                continue
            # read the raster band by band for this subset
            example_subset = np.nan_to_num(
                all_raster_bands[0].ReadAsArray(j*stride, i*stride, stride, stride))
            for band in all_raster_bands[1:]:
                example_subset = np.dstack((example_subset, np.nan_to_num(
                    band.ReadAsArray(j*stride, i*stride, stride, stride))))
            # save this example/label pair of numpy arrays as a pickle file with an index
            this_example_save_path = os.path.join(
                destination, '{}_{}_{}.pkl'.format(region, year, count))
            with open(this_example_save_path, 'wb') as this_pickle:
                pickle.dump((example_subset, label_subset),
                            file=this_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                print('log: Saved {} '.format(this_example_save_path))
                print(i*stride, (i+1)*stride, j*stride, (j+1)*stride)
            count += 1


def mask_landsat8_image_using_rasterized_shapefile(rasterized_shapefiles_path, district, this_landsat8_bands_list):
    this_shapefile_path = os.path.join(
        rasterized_shapefiles_path, "{}_shapefile.tif".format(district))
    ds = gdal.Open(this_shapefile_path)
    assert ds.RasterCount == 1
    shapefile_mask = np.array(ds.GetRasterBand(
        1).ReadAsArray(), dtype=np.uint8)
    clipped_full_spectrum = list()
    for idx, this_band in enumerate(this_landsat8_bands_list):
        print("{}: Band-{} Size: {}".format(district, idx, this_band.shape))
        clipped_full_spectrum.append(np.multiply(this_band, shapefile_mask))
    x_prev, y_prev = clipped_full_spectrum[0].shape
    x_fixed, y_fixed = int(128 * np.ceil(x_prev / 128)
                           ), int(128 * np.ceil(y_prev / 128))
    diff_x, diff_y = x_fixed - x_prev, y_fixed - y_prev
    diff_x_before, diff_y_before = diff_x // 2, diff_y // 2
    clipped_full_spectrum_resized = [np.pad(x, [(diff_x_before, diff_x - diff_x_before), (diff_y_before, diff_y - diff_y_before)], mode='constant')
                                     for x in clipped_full_spectrum]
    print("{}: Generated Image Size: {}".format(
        district, clipped_full_spectrum_resized[0].shape, len(clipped_full_spectrum_resized)))
    return clipped_full_spectrum_resized


def check_generated_dataset(path_to_dataset):
    for count in range(266):
        this_example_save_path = os.path.join(
            path_to_dataset, '{}.pkl'.format(count))
        with open(this_example_save_path, 'rb') as this_pickle:
            print('log: Reading {}'.format(this_example_save_path))
            (example_subset, label_subset) = pickle.load(
                this_pickle, encoding='latin1')
        show_image = np.asarray(
            255 * (example_subset[:, :, [4, 3, 2]] / 4096.0).clip(0, 1), dtype=np.uint8)
        plt.subplot(1, 2, 1)
        plt.imshow(show_image)
        plt.subplot(1, 2, 2)
        plt.imshow(label_subset)
        plt.show()


def check_generated_fnf_datapickle(example_path):
    with open(example_path, 'rb') as this_pickle:
        (example_subset, label_subset) = pickle.load(
            this_pickle, encoding='latin1')
        example_subset = np.nan_to_num(example_subset)
        label_subset = fix(np.nan_to_num(label_subset))
    this = np.asarray(255*(example_subset[:, :, [3, 2, 1]]), dtype=np.uint8)
    that = label_subset
    plt.subplot(121)
    plt.imshow(this)
    plt.subplot(122)
    plt.imshow(that)
    plt.show()


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

def main():
    # change these!
    data_directory_path = ...
    label_directory_path = ...
    destination = ...
    # generate pickle files to train from
    all_districts = ["abbottabad", "battagram", "buner", "chitral", "hangu", "haripur", "karak", "kohat", "kohistan",
                     "lower_dir", "malakand", "mansehra", "nowshehra", "shangla", "swat", "tor_ghar", "upper_dir"]
    # number of images generated depends on value of stride
    for district in all_districts:
        get_images_from_large_file(data_directory_path, label_directory_path, destination,
                                   bands=range(1, 12), year=2015, region=district, stride=256)


if __name__ == "__main__":
    main()
