#!/bin/bash


lr="1e-06"
topology="ENC_4_DEC_4"

bands="18"

echo " ************ Model 69 Test Results ************"
model="model_69_topology"$topology"_lr"$lr"_bands"$bands".pt"
echo $model

python train.py --function train_net --data /home/Projects/Forest/Data/generated_data --input_size 128 --workers 4 --data_split_lists output/training_lists/ --models_dir output/models/augmented --summary_dir output/PreTrainedModel --batch_size 16 --lr $lr --epochs 20 --log_after 10 --cuda 1 --device 0 --topology $topology --bands 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 --classes Non-Forest Forest --pretrained_model $model --error_maps_dir output/error_maps
