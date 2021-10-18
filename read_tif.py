
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


