import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.layers import Activation, MaxPool2D, Concatenate

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p 

def decoder_block(input, skip_features, num_filters, droupouts = 0.1):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = tf.keras.layers.Dropout(droupouts)(x)
    x = conv_block(x, num_filters)
    return x

def build_unet(num_classes, input_shape, droupouts = 0.1):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    p1 = tf.keras.layers.Dropout(droupouts)(p1)
    s2, p2 = encoder_block(p1, 128)
    p2 = tf.keras.layers.Dropout(droupouts)(p2)
    s3, p3 = encoder_block(p2, 256)
    p3 = tf.keras.layers.Dropout(droupouts)(p3)
    s4, p4 = encoder_block(p3, 512)
    p4 = tf.keras.layers.Dropout(droupouts)(p4)
    
    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    output = Conv2D(num_classes, (1,1), padding='same')(d4)

    outputs = Activation('softmax')(output)
    model = Model(inputs, outputs, name="U-Net")
    return model