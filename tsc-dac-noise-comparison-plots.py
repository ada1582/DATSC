#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 00:52:28 2020

@author: daniyalusmani1
"""


import pandas as pd
import argparse

import numpy as np
import matplotlib.pyplot as plt
import os


parser = argparse.ArgumentParser(description='Plot creation for TSC DAC noise level comparisons architecture',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--folder_list', default=None, type=str, nargs='+', help='list of folders with activation energies')
parser.add_argument('--noisy_levels', default=None, type=str, nargs='+', help='list of noise levels corresponding to folder list')
parser.add_argument('--class_number', default=None, type=int, help='number of class which for which plot needs to be generated')


args = parser.parse_args()

def _getListOfFilesForFolder(folderPath, activationType='train'):
    script_dir = os.path.dirname(os.path.realpath('__file__'))
   # script_dir = os.path.dirname(__file__) # <-- absolute dir the script is in
    abs_file_path = os.path.join(script_dir, folderPath)
    print(abs_file_path)
    entries = os.listdir(abs_file_path)
    entries = np.array(entries)
    indices = [activationType in entry for entry in entries]
    activationsFiles = entries[indices]
    return [(abs_file_path + '/' + file) for file in activationsFiles]

def _getClassActivationsForSelectedFolder(selectedClass, selectedFolder):
    trainFiles = _getListOfFilesForFolder(selectedFolder, 'train')
    valFiles = _getListOfFilesForFolder(selectedFolder, 'val')
    
    trainActivations = _extractAccumuatedActivations(trainFiles, selectedClass)
    valActivations = _extractAccumuatedActivations(valFiles, selectedClass)
    
    return trainActivations, valActivations
    

def _extractAccumuatedActivations(files, selectedClass):
    activations = []
    for file in files:
        epoch = np.load(file,  allow_pickle=True);
        if len(activations) == 0:
            activations = epoch.mean(axis=0);
            activations = np.expand_dims(activations, axis = 1)
        else:
            activations = np.append(activations, np.expand_dims(epoch.mean(axis=0), axis = 1), axis = 1)
        # print(activations.shape)
    classActivation = activations[selectedClass-1, :]
    return classActivation

def _generateActivationComparisionForSelectedClass(folderList, selectedClass):

    trainActivations = []
    valActivations = []
    for folder in folderList:
        trainFolderActivation, valFolderActivation =  _getClassActivationsForSelectedFolder(selectedClass, folder)
        trainFolderActivation = np.expand_dims(trainFolderActivation, axis = 1)
        valFolderActivation = np.expand_dims(valFolderActivation, axis = 1)
        if len(trainActivations) == 0:
            trainActivations = trainFolderActivation
            valActivations = trainFolderActivation
        else:
            
            trainActivations = np.append(trainActivations, trainFolderActivation[0:trainActivations.shape[0], : ], axis = 1)
            valActivations = np.append(valActivations, valActivations[0:valActivations.shape[0], : ], axis = 1)
    trainActivations = trainActivations.T
    valActivations = valActivations.T
    return trainActivations, valActivations

def _generateGraphForActivationType(activations, noiseLevels, activationType):
    for activation in zip(activations, noiseLevels):
        plt.plot(activation[0], label = '%s'%(activation[1]))
    if activationType == 'train':
        plt.title(' Comparison of activation energies for training')
    else:
        plt.title(' Comparison of activation energies for validation')
    plt.xlabel('Epochs')
    plt.ylabel('mean activation energy / epoch for class')
    plt.legend(loc='upper right', fontsize='xx-small')
    plt.show()     
    #plt.close()
    
def generateActivationComparisonPlotForFolderList():
    folderList = args.folder_list
    noiseLevels = args.noisy_levels
    classNum = args.class_number
    trainActivations, valActivations = _generateActivationComparisionForSelectedClass(folderList, classNum)
    _generateGraphForActivationType(trainActivations, noiseLevels, 'train')
    _generateGraphForActivationType(trainActivations, noiseLevels, 'val')
    
    
generateActivationComparisonPlotForFolderList() 
        
    

