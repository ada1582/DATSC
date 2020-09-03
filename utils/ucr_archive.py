# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import torch.utils.data as data
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import torch

class UCRArchive(data.Dataset):
    
    def __init__(self, archiveRoot, datasetName, datasetType = 'TRAIN', noise = 0, transform=None):
        self.samples = []
        self.labels = []
        dataset = [i for i in os.listdir(archiveRoot) if i == datasetName]
        if dataset:
            print('dataset is found')
            if noise == 0:
                data = pd.read_csv(archiveRoot + '/' + dataset[0] + '/' + dataset[0] + '_' + datasetType + '.tsv', sep='\t', header=None)
            else:
                data = pd.read_csv(archiveRoot + '/' + dataset[0] + '/' + dataset[0] + '_' + str(noise) + '_' + datasetType + '.tsv', sep='\t', header=None)
            self.labels = torch.Tensor(data.values[:, 0] - 1).long()
            self.targets = self.labels
            self.samples = data.drop(columns=[0]).to_numpy()
            self.data = self.samples
            
            std_ = self.samples.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0            
            self.samples = (self.samples - self.samples.mean(axis=1, keepdims=True)) / std_
            
        else:
            raise FileNotFoundError;

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        x =  np.expand_dims(x, axis=0)
        y = self.labels[idx]
        x = torch.Tensor(x)
        return x, y

