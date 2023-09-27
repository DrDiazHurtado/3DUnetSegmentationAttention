from hyperparam import *

import nibabel as nib
import numpy as np
import torchio as tio
import glob
import torch
import hyperparam
from hyperparam import *

# Import
import glob
import os
import shutil

# Starting the code
print("Creating patches")
if os.path.isdir(PATCHES_OUTPUT_DIR):
    for f in glob (os.path.join(PATCHES_OUTPUT_DIR,"*/*.nii*")):
        os.unlink (f)
    os.removedirs(PATCHES_OUTPUT_DIR)
os.mkdir(PATCHES_OUTPUT_DIR)
os.mkdir(PATCHES_OUTPUT_DIR+'images/')
os.mkdir(PATCHES_OUTPUT_DIR+'labels/')

images=sorted(glob.glob(INPUT_DIR+"images/*.nii"))
labels=sorted(glob.glob(INPUT_DIR+"labels/*.nii"))
print (INPUT_DIR+"images/*.nii")
subjects_list=[]
for i in range(len(images)):
    subjects_list.append(tio.Subject(image=tio.ScalarImage(images[i]),label=tio.ScalarImage(labels[i])))

# Transformations: not simply throw volumes into unet
rescale1000 = tio.RescaleIntensity(out_min_max=(0, 121000),exclude="labels") # Rescale only images 0 to 1000
#rescale1 = tio.RescaleIntensity(out_min_max=(0, 1),exclude="images") # Rescale only labels 0 to 1

# One of
anisotropy=tio.RandomAnisotropy() # anisotropy
blur=tio.RandomBlur() # Blur
noise=tio.RandomNoise() # Noise
bias=tio.RandomBiasField(coefficients=1)
motion=tio.RandomMotion(num_transforms=2, image_interpolation='nearest')
znormalization=tio.ZNormalization()
flip = tio.RandomFlip(axes=(0,1,2))

#transforms rescale + one of several
transformations = tio.Compose([
 #   tio.OneOf({ #anisotropy: 0.05,
               #blur: 0.05,
               #noise: 0.05 ,
               #bias: 0.05 ,
               #motion: 0.01,
            #flip : 0.3
  #          }),
              #znormalization,
            rescale1000
              # rescale1
])


subjects_dataset= tio.SubjectsDataset(subjects_list, transform= transformations)

patch_size=32
patch_overlap=4
queue_length = 4000

###############  Número de patches patch_size^3 que se van a sacar de cada array
samples_per_volume = 4000

sampler = tio.data.UniformSampler(patch_size=patch_size)

patches_queue = tio.Queue(
  subjects_dataset,
  queue_length,
  samples_per_volume,
  sampler,
  num_workers=1,
 )

patches_loader = torch.utils.data.DataLoader(
  patches_queue,
  batch_size=1,
  num_workers=0,  # this must be 0
  )

i=0
for patches_batch in patches_loader:
    inputs = patches_batch['image'][tio.DATA]
    targets = patches_batch['label'][tio.DATA]
    imagen= inputs.cpu().detach().numpy()
    etiqueta= targets.cpu().detach().numpy()
    np.save(PATCHES_OUTPUT_DIR+'images/'+str(i), imagen)
    np.save(PATCHES_OUTPUT_DIR+'labels/'+str(i), etiqueta)
    i=i+1
