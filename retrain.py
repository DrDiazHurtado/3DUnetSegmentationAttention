from hyperparam import *

import tensorflow
import losses
model2 = tensorflow.keras.models.load_model(WORKING_DIR+'modelo_escalado.h5',
                                            custom_objects={"tversky_loss":tversky_loss,
                                                            "dice_coef":dice_coef,
                                                           "tversky":tversky})
