from hyperparam import *


import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')

def conv3d_block(input_tensor,
                 n_filters,
                 kernel_size=3,
                 batchnorm=True,
                 strides=1,
                 dilation_rate=1,
                 recurrent=1):

    conv = Conv3D(filters=n_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  kernel_initializer="he_normal",
                  padding="same",
                  dilation_rate=dilation_rate)(input_tensor)
    if batchnorm:
        conv = BatchNormalization()(conv)
    output = LeakyReLU(alpha=alpha)(conv)

    for _ in range(recurrent - 1):
        conv = Conv3D(filters=n_filters,
                      kernel_size=kernel_size,
                      strides=1,
                      kernel_initializer="he_normal",
                      padding="same",
                      dilation_rate=dilation_rate)(output)
        if batchnorm:
            conv = BatchNormalization()(conv)
        res = LeakyReLU(alpha=alpha)(conv)
        output = Add()([output, res])

    return output

def residual_block(input_tensor,
                   n_filters,
                   kernel_size=3,
                   strides=1,
                   batchnorm=True,
                   recurrent=1,
                   dilation_rate=1):

    res = conv3d_block(input_tensor,
                       n_filters=n_filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       batchnorm=batchnorm,
                       dilation_rate=dilation_rate,
                       recurrent=recurrent)
    res = conv3d_block(res,
                       n_filters=n_filters,
                       kernel_size=kernel_size,
                       strides=1,
                       batchnorm=batchnorm,
                       dilation_rate=dilation_rate,
                       recurrent=recurrent)

    shortcut = conv3d_block(input_tensor,
                            n_filters=n_filters,
                            kernel_size=1,
                            strides=strides,
                            batchnorm=batchnorm,
                            dilation_rate=1)
    if batchnorm:
        shortcut = BatchNormalization()(shortcut)

    output = Add()([shortcut, res])
    return output


def inception_block(input_tensor,
                    n_filters,
                    kernel_size=3,
                    strides=1,
                    batchnorm=True,
                    recurrent=1,
                    layers=[]):

    res = conv3d_block(input_tensor,
                       n_filters=n_filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       batchnorm=batchnorm,
                       dilation_rate=1,
                       recurrent=recurrent)

    temp = []
    for layer in layers:
        local_res = res
        for conv in layer:
            incep_kernel_size = conv[0]
            incep_dilation_rate = conv[1]
            local_res = conv3d_block(local_res,
                                     n_filters=n_filters,
                                     kernel_size=incep_kernel_size,
                                     strides=1,
                                     batchnorm=batchnorm,
                                     dilation_rate=incep_dilation_rate,
                                     recurrent=recurrent)
        temp.append(local_res)

    temp = concatenate(temp)
    res = conv3d_block(temp,
                       n_filters=n_filters,
                       kernel_size=1,
                       strides=1,
                       batchnorm=batchnorm,
                       dilation_rate=1)

    shortcut = conv3d_block(input_tensor,
                            n_filters=n_filters,
                            kernel_size=1,
                            strides=strides,
                            batchnorm=batchnorm,
                            dilation_rate=1)
    if batchnorm:
        shortcut = BatchNormalization()(shortcut)

    output = Add()([shortcut, res])
    return output

def transpose_block(input_tensor,
                    skip_tensor,
                    n_filters,
                    kernel_size=3,
                    strides=1,
                    batchnorm=True,
                    recurrent=1):

    shape_x = K.int_shape(input_tensor)
    shape_xskip = K.int_shape(skip_tensor)

    conv = Conv3DTranspose(filters=n_filters,
                           kernel_size=kernel_size,
                           padding='same',
                           strides=(shape_xskip[1] // shape_x[1],
                                    shape_xskip[2] // shape_x[2],
                                    shape_xskip[3] // shape_x[3]),
                           kernel_initializer="he_normal")(input_tensor)
    conv = LeakyReLU(alpha=alpha)(conv)

    act = conv3d_block(conv,
                       n_filters=n_filters,
                       kernel_size=kernel_size,
                       strides=1,
                       batchnorm=batchnorm,
                       dilation_rate=1,
                       recurrent=recurrent)
    output = Concatenate(axis=4)([act, skip_tensor])
    return output

def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4),
                       arguments={'repnum': rep})(tensor)
    return my_repeat


def AttnGatingBlock(x, g, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)
    phi_g = Conv3D(filters=inter_shape,
                   kernel_size=1,
                   strides=1,
                   padding='same')(g)
    theta_x = Conv3D(filters=inter_shape,
                     kernel_size=3,
                     strides=(shape_x[1] // shape_g[1],
                              shape_x[2] // shape_g[2],
                              shape_x[3] // shape_g[3]),
                     padding='same')(x)
    add_xg = add([phi_g, theta_x])
    add_xg = Activation('relu')(add_xg)

    psi = Conv3D(filters=1, kernel_size=1, padding='same')(add_xg)
    psi = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(psi)

    upsample_sigmoid_xg = UpSampling3D(
        size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2],
              shape_x[3] // shape_sigmoid[3]))(psi)

    upsample_sigmoid_xg = expend_as(upsample_sigmoid_xg, shape_x[4])

    attn_coefficients = multiply([upsample_sigmoid_xg, x])
    output = Conv3D(filters=shape_x[4],
                    kernel_size=1,
                    strides=1,
                    padding='same')(attn_coefficients)
    output = BatchNormalization()(output)
    return output


def GatingSignal(input_tensor, batchnorm=True):
    shape = K.int_shape(input_tensor)
    conv = Conv3D(filters=shape[4],
                  kernel_size=1,
                  strides=1,
                  padding="same",
                  kernel_initializer="he_normal")(input_tensor)
    if batchnorm:
        conv = BatchNormalization()(conv)
    output = LeakyReLU(alpha=alpha)(conv)
    return output

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D

def network(input_img, n_filters=16, batchnorm=True):
    c0 = inception_block(input_img,
                         n_filters=n_filters,
                         batchnorm=batchnorm,
                         strides=1,
                         recurrent=2,
                         layers=[[(3, 1), (3, 1)], [(3, 2)]])  # 512x512x512
    c1 = inception_block(c0,
                         n_filters=n_filters * 2,
                         batchnorm=batchnorm,
                         strides=1,
                         recurrent=2,
                         layers=[[(3, 1), (3, 1)], [(3, 2)]])  # 256x256x256
    c2 = inception_block(c1,
                         n_filters=n_filters * 4,
                         batchnorm=batchnorm,
                         strides=1,
                         recurrent=2,
                         layers=[[(3, 1), (3, 1)], [(3, 2)]])  # 128x128x128

    c3 = inception_block(c2,
                         n_filters=n_filters * 8,
                         batchnorm=batchnorm,
                         strides=1,
                         recurrent=2,
                         layers=[[(3, 1), (3, 1)], [(3, 2)]])  # 64x64x64
    b0 = inception_block(c3,
                         n_filters=n_filters * 16,
                         batchnorm=batchnorm,
                         strides=1,
                         recurrent=2,
                         layers=[[(3, 1), (3, 1)], [(3, 2)]])  # 32x32x32
    attn0 = AttnGatingBlock(c3, b0, n_filters * 16)
    u0 = transpose_block(b0,
                         attn0,
                         n_filters=n_filters * 8,
                         batchnorm=batchnorm,
                         recurrent=2)  # 64x64x64
    attn1 = AttnGatingBlock(c2, u0, n_filters * 8)
    u1 = transpose_block(u0,
                         attn1,
                         n_filters=n_filters * 4,
                         batchnorm=batchnorm,
                         recurrent=2)  # 128x128x128
    attn2 = AttnGatingBlock(c1, u1, n_filters * 4)
    u2 = transpose_block(u1,
                         attn2,
                         n_filters=n_filters * 2,
                         batchnorm=batchnorm,
                         recurrent=2)  # 256x256x256
    u3 = transpose_block(u2,
                         c0,
                         n_filters=n_filters,
                         batchnorm=batchnorm,
                         recurrent=2)  # 512x512x512
    outputs = Conv3D(filters=1, kernel_size=1, strides=1,
                     activation='sigmoid')(u3)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
