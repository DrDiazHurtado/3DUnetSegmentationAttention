from hyperparam import *

# Import
import glob
import os
import sys
from tensorflow.keras.layers import Input
from network import *
from tensorflow.keras.optimizers import Adam
import tensorflow
import shutil
import nibabel as nib


# Import locals
from hyperparam import *
from patch import *
from callbacks import *
from datagen import *
from losses import *
from network import *
from predict import *

print("Creating patches")

os.system(PYTHON_BIN+'python3.8 '+WORKING_DIR+'patch.py')

print("Patches done")

# Train

images_list=sorted(glob.glob(os.path.join(images_path,'*.npy')))
labels_list=sorted(glob.glob(os.path.join(labels_path,'*.npy')))

list_IDs=[{'image': image, 'label':label} for image, label in zip(images_list, labels_list)]
input_img = Input(input_dimensions)

model = network(input_img, n_filters=num_initial_filters, batchnorm=batchnorm)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=metrics, run_eagerly=True)

print("Creating model")

train_gen = DataGenerator(list_IDs, dim=dimensions, batch_size=batch_size, shuffle=True)

print("Model fit")
model.fit( train_gen, steps_per_epoch=steps_per_epoch,callbacks = callbacks, epochs=epochs , verbose=2, workers=1,use_multiprocessing=True)
print("Model save")
tensorflow.keras.models.save_model(model, WORKING_DIR+'modelos/modelo_escalado.h5')

# Predict

print("Creating predict")

for patches_batch in patch_loader:
    input_tensor = patches_batch['image'][tio.DATA].detach().numpy() # 1,1,32,32,32
    locations = patches_batch[tio.LOCATION]
    np1=input_tensor[0,:,:,:] # (1,32,32,32)
    np1=np.moveaxis(np1, 0, 3) # 32,32,32,1
    logits = model(np1)
    logits=logits.numpy()
    np2=np.moveaxis(logits,4,0)
    np2=np.moveaxis(np2,4,1)
    torch_output=torch.from_numpy(np2)
    aggregator.add_batch(torch_output, locations)

output_tensor = aggregator.get_output_tensor()

salida=output_tensor.detach().numpy()
salida2=salida[0,:,:,:] # Leave 181x217x181
new_image = nib.Nifti1Image(salida2, affine=np.eye(4))
nib.save(new_image, WORKING_DIR+'output-raw.nii')
mask=subjects_list[0]['label'].data.detach().numpy()
final1=dice_coef(salida , mask).numpy()
final2=dice_coef(salida2, mask).numpy()
print("Dice raw:", final1, "Dice dicotomized:", final2)
