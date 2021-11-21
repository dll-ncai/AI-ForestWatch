# Copyright (c) 2021, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env bash
for year in `seq 2016 2017`;
	do
	for region in `seq 1 4`;
        	do
                	# $i
			python inference.py -m '/home/annus/Desktop/palsar/palsar_models_focal/trained_separately/train_on_2015/model-4.pt' -b 20 -y $year -r $region
	        done
	done
