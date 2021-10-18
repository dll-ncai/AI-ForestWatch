lr="1e-06"
topology="ENC_4_DEC_4"
bands="18"

model="/home/Projects/Forest/AnasWork/BTT-ForestCoverChange/LandCoverSegmentation/semantic_segmentation/output/models/augmented/model_69_topologyENC_4_DEC_4_lr1e-06_bands18.pt"
echo $model

python inference.py --data_path /home/Projects/Forest/Data/all_billion_tree_regions/landsat-8/test_data/ --shapefiles /home/Projects/Forest/Data/District_Shapefiles_as_Clipping_bands-20210913T193233Z-001/District_Shapefiles_as_Clipping_bands --batch_size 32 --cuda 1 --device 0 --topology $topology --bands 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 --classes Non-Forest Forest --model $model --destination /home/Projects/Forest/Data/dest
