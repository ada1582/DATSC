python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/Crop/Crop_TRAIN.tsv --dataset Crop --noisy_dataset data/UCRArchive_2018/Crop/Crop --noisy_percentage 0.3
python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/Crop/Crop_TRAIN.tsv --dataset Crop --noisy_dataset data/UCRArchive_2018/Crop/Crop --noisy_percentage 0.5
python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/Crop/Crop_TRAIN.tsv --dataset Crop --noisy_dataset data/UCRArchive_2018/Crop/Crop --noisy_percentage 0.75

python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/SmoothSubspace/SmoothSubspace_TRAIN.tsv --dataset SmoothSubspace --noisy_dataset data/UCRArchive_2018/SmoothSubspace/SmoothSubspace --noisy_percentage 0.3
python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/SmoothSubspace/SmoothSubspace_TRAIN.tsv --dataset SmoothSubspace --noisy_dataset data/UCRArchive_2018/SmoothSubspace/SmoothSubspace --noisy_percentage 0.5
python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/SmoothSubspace/SmoothSubspace_TRAIN.tsv --dataset SmoothSubspace --noisy_dataset data/UCRArchive_2018/SmoothSubspace/SmoothSubspace --noisy_percentage 0.75

python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/Chinatown/Chinatown_TRAIN.tsv --dataset Chinatown --noisy_dataset data/UCRArchive_2018/Chinatown/Chinatown --noisy_percentage 0.3
python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/Chinatown/Chinatown_TRAIN.tsv --dataset Chinatown --noisy_dataset data/UCRArchive_2018/Chinatown/Chinatown --noisy_percentage 0.5
python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/Chinatown/Chinatown_TRAIN.tsv --dataset Chinatown --noisy_dataset data/UCRArchive_2018/Chinatown/Chinatown --noisy_percentage 0.75


python3 tsc-dac-plots.py --dataset crop --exp_name 2-lstm-simple-no-noise

python3 tsc-dac-plots.py --dataset crop --exp_name 2-lstm-simple-0.3-epoch-300-learning-100

python3 tsc-dac-plots.py --dataset crop --exp_name 2-lstm-simple-0.3-epoch-300-learning-100-dac-learning_rate-60_1-120_2

python3 tsc-dac-plots.py --dataset ai_crop --exp_name 2-lstm-dac-epoch-300-learning-100-learning_rate-0.05-60_1-120_2

python3 tsc-dac-plots.py --dataset ai_crop --exp_name balanced-imputed-2015-1-lstm-dac-epoch-100-learning-50-learning_rate-0.05-60_1-120_2
