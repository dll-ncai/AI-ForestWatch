# Copyright (c) 2021, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

np.random.seed(123)

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


def get_dataloaders_generated_data(generated_data_path, data_split_lists_path, model_input_size, bands, num_classes, train_split, one_hot, batch_size,
                                   num_workers):
    class dataset(Dataset):
        def __init__(self, data_list, data_map_path, stride, mode='train', transformation=None):
            super(dataset, self).__init__()
            self.model_input_size = model_input_size
            self.data_list = data_list
            self.all_images = []
            self.total_images = 0
            self.stride = stride
            self.one_hot = one_hot
            self.bands = [x-1 for x in bands]
            self.num_classes = num_classes
            self.transformation = transformation
            self.mode = mode
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
            ndvi_band = (this_example_subset[:,:,4]-this_example_subset[:,:,3])/(this_example_subset[:,:,4]+this_example_subset[:,:,3]+1e-7)
            evi_band = 2.5*(this_example_subset[:,:,4]-this_example_subset[:,:,3])/(this_example_subset[:,:,4]+6*this_example_subset[:,:,3]-7.5*this_example_subset[:,:,1]+1)
            savi_band = 1.5*(this_example_subset[:,:,4]-this_example_subset[:,:,3])/(this_example_subset[:,:,4]+this_example_subset[:,:,3]+0.5)
            msavi_band = 0.5*(2*this_example_subset[:,:,4]+1-np.sqrt((2*this_example_subset[:,:,4]+1)**2-8*(this_example_subset[:,:,4]-this_example_subset[:,:,3])))
            ndmi_band = (this_example_subset[:,:,4]-this_example_subset[:,:,5])/(this_example_subset[:,:,4]+this_example_subset[:,:,5]+1e-7)
            nbr_band = (this_example_subset[:,:,4]-this_example_subset[:,:,6])/(this_example_subset[:,:,4]+this_example_subset[:,:,6]+1e-7)
            nbr2_band = (this_example_subset[:,:,5]-this_example_subset[:,:,6])/(this_example_subset[:,:,5]+this_example_subset[:,:,6]+1e-7)
            this_example_subset = np.dstack((this_example_subset, np.nan_to_num(ndvi_band)))
            this_example_subset = np.dstack((this_example_subset, np.nan_to_num(evi_band)))
            this_example_subset = np.dstack((this_example_subset, np.nan_to_num(savi_band)))
            this_example_subset = np.dstack((this_example_subset, np.nan_to_num(msavi_band)))
            this_example_subset = np.dstack((this_example_subset, np.nan_to_num(ndmi_band)))
            this_example_subset = np.dstack((this_example_subset, np.nan_to_num(nbr_band)))
            this_example_subset = np.dstack((this_example_subset, np.nan_to_num(nbr2_band)))
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
    ######################################################################################
    transformation = None
    train_list, eval_list, test_list, temp_list = [], [], [], []
    if not os.path.exists(data_split_lists_path):
        print('LOG: No saved data found. Making new data directory {}'.format(data_split_lists_path))
        os.mkdir(data_split_lists_path)
        
    extended_data_path = generated_data_path
    full_examples_list = [os.path.join(extended_data_path, x) for x in os.listdir(extended_data_path)]
    random.shuffle(full_examples_list)
    train_split = int(train_split*len(full_examples_list))
    train_list = full_examples_list[:train_split]
    temp_list = full_examples_list[train_split:]
    eval_list = temp_list[0:len(temp_list)//2]
    test_list = temp_list[len(temp_list)//2:]
    ######################################################################################
    print('LOG: [train_list, eval_list, test_list] ->', len(train_list), len(eval_list), len(test_list))
    print('LOG: set(train_list).isdisjoint(set(eval_list)) ->', set(train_list).isdisjoint(set(eval_list)))
    print('LOG: set(train_list).isdisjoint(set(test_list)) ->', set(train_list).isdisjoint(set(test_list)))
    print('LOG: set(test_list).isdisjoint(set(eval_list)) ->', set(test_list).isdisjoint(set(eval_list)))
    # create dataset class instances
    # images_per_image means approx. how many images are in each example
    train_data = dataset(data_list=train_list, data_map_path=os.path.join(data_split_lists_path, 'train_datamap.pkl'), mode='train', stride=8,
                         transformation=transformation)  # more images for training
    eval_data = dataset(data_list=eval_list, data_map_path=os.path.join(data_split_lists_path, 'eval_datamap.pkl'), mode='test', stride=model_input_size,
                        transformation=transformation)
    test_data = dataset(data_list=test_list, data_map_path=os.path.join(data_split_lists_path, 'test_datamap.pkl'), mode='test', stride=model_input_size,
                        transformation=transformation)
    print('LOG: [train_data, eval_data, test_data] ->', len(train_data), len(eval_data), len(test_data))
    print('LOG: Data Split Integrity: set(train_list).isdisjoint(set(eval_list)) ->', set(train_list).isdisjoint(set(eval_list)))
    print('LOG: Data Split Integrity: set(train_list).isdisjoint(set(test_list)) ->', set(train_list).isdisjoint(set(test_list)))
    print('LOG: Data Split Integrity: set(test_list).isdisjoint(set(eval_list)) ->', set(test_list).isdisjoint(set(eval_list)))
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataloader


def main():
    loaders = get_dataloaders_generated_data(generated_data_path='/mnt/e/Forest Cover - Redo 2020/Google Cloud - Training/Training Data/Clipped dataset/'
                                                                 'Pickled_data/',
                                             save_data_path="/mnt/e/Forest Cover - Redo 2020/Google Cloud - Training/training_lists",
                                             model_input_size=128, num_classes=2, train_split=0.8, one_hot=True, batch_size=16, num_workers=4, max_label=2)

    train_dataloader, val_dataloader, test_dataloader = loaders
    while True:
        for idx, data in enumerate(train_dataloader):
            examples, labels = data['input'], data['label']
            print('-> on batch {}/{}, {}'.format(idx+1, len(train_dataloader), examples.size()))
            example_subset = (examples[0].numpy()).transpose(1, 2, 0)
            for j in range(7):
                plt.subplot(4,3,j+1)
                plt.imshow(example_subset[:,:,11+j])
            plt.show()


if __name__ == '__main__':
    main()
