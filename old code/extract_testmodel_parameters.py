# Copyright (c) 2021, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import re
import csv
import itertools


if len(sys.argv) < 2:
	print("pass a file via command line to extract data")
	exit(1)

filename = sys.argv[-1]

output_filename = "result_"+filename+".csv"

print(output_filename)
#print(filename)



#search_bands = "bands ="
#search_lr    = "lr ="
search_model = "Test Results"
search_test = "log: test:: "
search_accuracy  = "[LOG] Net"

search_non_forest = "  Non-Forest "
search_forest = "  Forest "


data_list = []
comp_list = []
#bands_list = []
#lr_list = []

with open (filename, 'rt') as myfile:
	for line in myfile:
		if search_model in line:
			data_r = line
			#print(modelno)
			modelno = re.findall('\d+', data_r)
			#data_list = []
			comp_list.append(data_list)
			data_list = []
			data_list = data_list + modelno
			#print(comp_list)
			#exit()
			
		elif search_test in line:
			data_r = line 
			testdata = re.findall('\d+\.\d+', data_r)
			#print(testdata)
			data_list = data_list + testdata
		elif search_accuracy in line:
			data_r = line
			#print(data_r)
			modelno = re.findall('\d+\.\d+', data_r)
			data_list = data_list + modelno
			#print(modelno)
			#exit()
			
		elif search_non_forest in line:
			data_r = line
			non_forest = re.findall('\d+\.\d+', data_r)
			data_list = data_list + non_forest 
		elif search_forest in line:
			#extracted_data.append(line)
			data_r = line
			forest = re.findall('\d+\.\d+', data_r)
			#forest_list.append(forest)
			data_list = data_list + forest
			#print(data_list)
			#exit()
			#comp_list.append(data_list)
			#data_list = []
			#exit()
	#	comp_list.append(data_list)

del comp_list[0]
#print(comp_list)

fields = ['Model no', 'test_loss_all', 'test_accuracy_all', 'NonForest_Precision_all', 'NonForest_recall_all', 'NonForest_f1score_all','Forest_Precision_all', 'Forest_recall_all', 'Forest_f1score_all', 'test_accuracy_WO_CaUD', 'NonForest_Precision_WO_CaUD', 'NonForest_recall_WO_CaUD', 'NonForest_f1score_WO_CaUD','Forest_Precision_WO_CaUD', 'Forest_recall_WO_CaUD', 'Forest_f1score_WO_CaUD']

with open(output_filename, 'w') as csvfile:
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(fields)
	csvwriter.writerows(comp_list)
