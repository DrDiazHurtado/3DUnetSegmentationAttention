from hyperparam import *

import numpy as np
import tensorflow
import glob
import torch
import torch.nn as nn
import torchio as tio
from patch import rescale1000
import nibabel as nib

patch_overlap = 4, 4, 4  # or just 4
patch_size = 32, 32, 32

images=sorted(glob.glob(INPUT_DIR+"/images/*.nii"))
labels=sorted(glob.glob(INPUT_DIR+"/labels/*.nii"))
subjects_list=[]
for i in range(len(images)):
    subjects_list.append(tio.Subject(image=tio.ScalarImage(images[i]),label=tio.ScalarImage(labels[i])))

subjects_dataset= tio.SubjectsDataset(subjects_list,transform= rescale1000 ) # transform= None

# Cargamos el caso 2 mediante grid_sampler con batch_size=1

grid_sampler = tio.inference.GridSampler(subjects_list[1],patch_size, patch_overlap)
patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
aggregator = tio.inference.GridAggregator(grid_sampler)
