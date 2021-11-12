# Copyright (c) 2021, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE.md file in the root directory of this source tree.

from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as pl
from scipy import misc
import numpy as np
import imageio


def main():
    this = imageio.imread('/home/Projects/Forest/Data/all_billion_tree_regions/landsat-8/train_data/landsat8_2015_region_swat.tif')
    print(np.max(this))
    for i in range(100):
        rand = np.random.randint(0,this.shape[0]-64)
        new = this[rand:rand+64,rand:rand+64,:4]
        pl.imshow(new)
        pl.show()
    pass


if __name__ == '__main__':
    main()


this = imageio.imread('/home/Projects/Forest/Data/all_billion_tree_regions/landsat-8/train_data/landsat8_2015_region_swat.tif')


