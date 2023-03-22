#!/bin/bash

declare -a arr=(
	#"combined_train_xnor_resnet14.conf"
	#"combined_train_fp_resnet10.conf"
	#"combined_train_float_resnet6.conf"
	"combined_train_float_resnet8a.conf"
	"combined_train_float_resnet8b.conf"
	#"combined_train_float_resnet10.conf"
	#"combined_train_float_resnet14.conf"
	#"combined_train_float_resnet18.conf"
)

for i in "${arr[@]}"
do
   python train_combined.py --validate --conf $i #combined_train_xnor_resnet18.conf
done
