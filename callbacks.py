from hyperparam import *

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TerminateOnNaN
import tensorflow.keras.backend as K
import glob
import os


################################################################################
# Compiling
################################################################################

callbacks = [
   # EarlyStopping(monitor='dice_coef', patience=3, verbose=2),
    ModelCheckpoint(filepath='/home/noorm/Escritorio/AttUnetModel/model.{epoch:02d}-{loss:.2f}.h5',
                    monitor='dice_coef',
                    mode='max',
                    verbose=1,
                    save_freq= int(50 * steps_per_epoch),
                    save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir='/home/noorm/Escritorio/AttUnetModel/logs',write_graph=True),
    TerminateOnNaN()
]
