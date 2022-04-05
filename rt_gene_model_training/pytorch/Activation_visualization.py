import numpy as np
import torch
import os
from evaluate_model import rename_state_dict
from functools import partial
from rt_gene.src.rt_gene.gaze_estimation_models_pytorch import GazeEstimationModelResnet18, \
    GazeEstimationModelVGG, GazeEstimationModelPreactResnet
from utils.GazeAngleAccuracy import GazeAngleAccuracy
import h5py
from rtgene_dataset import RTGENEH5Dataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    weight_location = '/media/grant/Dataset/Checkpoints/RTGene_network_trained_on_MPII/fold_0'
    weight_name = 'epoch=14-val_loss=0.006.ckpt'
    dataset_location = '/home/grant/Gaze_Dataset'
    dataset_name = 'mpii_dataset.hdf5'
    weight_file = os.path.join(weight_location, weight_name)
    dataset_file = os.path.join(dataset_location, dataset_name)
    weight_file = torch.load(weight_file)['state_dict']
    weight_file = rename_state_dict(weight_file)

    model = partial(GazeEstimationModelVGG, num_out=2)()
    model.load_state_dict(weight_file)
    model.eval()
    test_subject = [1]
    criterion = GazeAngleAccuracy()

    dataset = h5py.File(dataset_file, mode='r')
    data_test = RTGENEH5Dataset(dataset, subject_list=test_subject)
    data_loader = DataLoader(data_test, batch_size=64, shuffle=True, pin_memory=False)



