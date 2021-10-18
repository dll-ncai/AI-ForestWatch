#!/bin/bash

lr="1e-06"
topology="ENC_4_DEC_4"
bands="18"

model="/home/Projects/Forest/AnasWork/BTT-ForestCoverChange/LandCoverSegmentation/semantic_segmentation/output/models/augmented/model_69_topologyENC_4_DEC_4_lr1e-06_bands18.pt"
echo $model

python inference_final.py --m $model --d /home/Projects/Forest/Data/all_billion_tree_regions/landsat-8/test_data --s /home/Projects/Forest/Data/dest --b 64 --cuda 0 --device 0

