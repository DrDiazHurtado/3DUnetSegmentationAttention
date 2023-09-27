from losses import dice_coef, tversky, tversky_loss

WORKING_DIR='/data/'
INPUT_DIR=WORKING_DIR+'ISBIFlairUnion/'
PATCHES_OUTPUT_DIR=WORKING_DIR+'NewPatches/'
PYTHON_BIN="/opt/conda/bin/"

checkpoint_path = WORKING_DIR
log_path = WORKING_DIR
save_path = WORKING_DIR
train_path = WORKING_DIR
test_path = WORKING_DIR
images_path=PATCHES_OUTPUT_DIR+"images"
labels_path= PATCHES_OUTPUT_DIR+"labels"
checkpoint_path= WORKING_DIR

alpha = 0.1
input_dimensions = (32, 32, 32, 1)
dimensions = (32, 32, 32)
num_initial_filters = 4
batchnorm = True
num_gpu = 0
batch_size = 10
steps_per_epoch = 10
learning_rate = 0.01
loss = tversky_loss
metrics = [dice_coef,tversky]
epochs = 50
shuffle=True

patch_size=32
patch_overlap=4
queue_length = 4000
samples_per_volume = 4000
