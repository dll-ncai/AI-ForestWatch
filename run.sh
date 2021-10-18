
source activate py2.7





python train.py --function generate_error_maps --data home/kzm/Khuzaeymah/Summer21/training_2015_pickled_data/ --input_size 128 --workers 4 --data_split_lists /work/mohsin/BTT_districts_maps/output/training_lists/ --models_dir /work/mohsin/BTT_districts_maps/output/models/augmented/ --summary_dir /work/mohsin/BTT_districts_maps/output/summary/augmented/ --batch_size 64 --lr  1e-6 --epochs 50 --log_after 10 --cuda 0 --device 0 --topology ENC_4_DEC_4 --bands 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 --classes Non-Forest Forest --pretrained_model /work/mohsin/BTT_districts_maps/output/models/augmented/model_53_topologyENC_4_DEC_4_lr1e-06_bands18.pt   --error_maps_dir '/work/mohsin/BTT_districts_maps/output/error_maps'



## RGB


#python train.py --function generate_error_maps --data /work/mohsin/BTT_districts_maps/training_2015_pickled_data/ --input_size 128 --workers 4 --data_split_lists /work/mohsin/BTT_districts_maps/output/training_lists/ --models_dir /work/mohsin/BTT_districts_maps/output/models/rgb/ --summary_dir /work/mohsin/BTT_districts_maps/output/summary/augmented/ --batch_size 64 --lr  1e-6 --epochs 50 --log_after 10 --cuda 0 --device 0 --topology ENC_4_DEC_4 --bands 2 3 4 --classes Non-Forest Forest --pretrained_model /work/mohsin/BTT_districts_maps/output/models/rgb/   --error_maps_dir '/work/mohsin/BTT_districts_maps/output/error_maps'
#python train.py --function eval_net --data /work/mohsin/BTT_districts_maps/training_2015_pickled_data/ --input_size 128 --workers 4 --data_split_lists /work/mohsin/BTT_districts_maps/output/training_lists/ --models_dir /work/mohsin/BTT_districts_maps/output/models/rgb/ --summary_dir /work/mohsin/BTT_districts_maps/output/summary/rgb/ --batch_size 64 --lr $lr --epochs 50 --log_after 10 --cuda 0 --device 0 --topology ENC_4_DEC_4 --bands 2 3 4 --classes Non-Forest Forest --pretrained_model $model


## FS

#python train.py --function generate_error_maps --data /work/mohsin/BTT_districts_maps/training_2015_pickled_data/ --input_size 128 --workers 4 --data_split_lists /work/mohsin/BTT_districts_maps/output/training_lists/ --models_dir /work/mohsin/BTT_districts_maps/output/models/augmented/ --summary_dir /work/mohsin/BTT_districts_maps/output/summary/augmented/ --batch_size 64 --lr  1e-6 --epochs 50 --log_after 10 --cuda 0 --device 0 --topology ENC_4_DEC_4 --bands 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 --classes Non-Forest Forest --pretrained_model /work/mohsin/BTT_districts_maps/output/models/augmented/model_53_topologyENC_4_DEC_4_lr1e-06_bands18.pt   --error_maps_dir '/work/mohsin/BTT_districts_maps/output/error_maps'


## indices

#python train.py --function generate_error_maps --data /work/mohsin/BTT_districts_maps/training_2015_pickled_data/ --input_size 128 --workers 4 --data_split_lists /work/mohsin/BTT_districts_maps/output/training_lists/ --models_dir /work/mohsin/BTT_districts_maps/output/models/augmented/ --summary_dir /work/mohsin/BTT_districts_maps/output/summary/augmented/ --batch_size 64 --lr  1e-6 --epochs 50 --log_after 10 --cuda 0 --device 0 --topology ENC_4_DEC_4 --bands 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 --classes Non-Forest Forest --pretrained_model /work/mohsin/BTT_districts_maps/output/models/augmented/model_53_topologyENC_4_DEC_4_lr1e-06_bands18.pt   --error_maps_dir '/work/mohsin/BTT_districts_maps/output/error_maps'



