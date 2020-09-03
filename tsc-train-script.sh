#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 01:48:36 2020

@author: daniyalusmani1
"""

#simple without experiment name
#python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-resnet --noise_percentage 0.3 --epochs 100 --loss_fn dac_loss --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0

# with experiment name
#python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 3 --epochs 5 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 3-lstm-simple-no-noise-epoch-500 --lr 0.9 --output_path results/crop/


python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 2-lstm-dac-0.3-epoch-300 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.9 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 2-lstm-simple-0.5-epoch-300 --noise_percentage 0.5 --lr 0.9 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 2-lstm-simple-0.75-epoch-300 --noise_percentage 0.75 --lr 0.9 --output_path results/crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 3 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 3-lstm-simple-0.3-epoch-300 --noise_percentage 0.3 --lr 0.9 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 3 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 3-lstm-simple-0.5-epoch-300 --noise_percentage 0.5 --lr 0.9 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 3 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 3-lstm-simple-0.75-epoch-300 --noise_percentage 0.75 --lr 0.9 --output_path results/crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 2-lstm-simple-0.3-epoch-300-learning-100-dac --loss_fn dac_loss --noise_percentage 0.3 --learn_epochs 100 --lr 0.9 --output_path results/crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 2-lstm-simple-0.3-epoch-300-learning-100-dac-learning_rate-60_1-120_2 --loss_fn dac_loss --noise_percentage 0.3 --learn_epochs 100 --lr 0.9 --output_path results/crop/

#ai_crop script
python3 tsc-train.py --datadir data/ai_crop --dataset crop_tsc_balanced_filled_2015.csv --nesterov --net_type tsc-lstm --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 2-lstm-dac-epoch-300-learning-100-learning_rate-0.01-60_1-120_2 --loss_fn dac_loss --learn_epochs 100 --lr 0.05 --output_path results/ai_crop/


python3 tsc-train.py --datadir data/ai_crop --dataset crop_tsc_balanced_imputed_2015.csv --nesterov --net_type tsc-resnet --depth 1 --epochs 500 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name balanced-imputed-2015-tsc-resnet-dac-epoch-500-learning-300-learning_rate-0.1-1_300-2_350-3_400-4_450 --loss_fn dac_loss --learn_epochs 300 --lr 0.1 --output_path results/ai_crop/

#ai_crop pca dataset 
python3 tsc-train.py --datadir data/ai_crop --dataset crop_tsc_balanced_imputed_PCA_2015.csv --nesterov --net_type tsc-resnet --depth 1 --epochs 500 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name balanced-imputed-PCA-2015-tsc-resnet-dac-epoch-500-learning-300-learning_rate-0.1-1_300-2_350-3_400-4_450 --loss_fn dac_loss --learn_epochs 300 --lr 0.1 --output_path results/ai_crop/

python3 tsc-train.py --datadir data/ai_crop --dataset crop_tsc_balanced_imputed_PCA_2015.csv --nesterov --net_type tsc-lstm --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name balanced-imputed-PCA-2015-2-lstm-dac-epoch-300-learning-50-learning_rate-0.00000001-60_1-120_2 --loss_fn dac_loss --learn_epochs 50 --lr 0.00000001 --output_path results/ai_crop/