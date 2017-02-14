from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Lambda, Dropout, merge

def encoder_decoder(input_shape):
    return Sequential([
        Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape),
        Convolution2D(1, 1, 1),
        Convolution2D(8, 3, 3, activation='elu', border_mode='same'),
        Convolution2D(8, 3, 3, activation='elu', border_mode='same'),
        MaxPooling2D((2,2), strides=(2,2)),
        Dropout(0.5),
        Convolution2D(16, 3, 3, activation='elu', border_mode='same'),
        Convolution2D(16, 3, 3, activation='elu', border_mode='same'),
        MaxPooling2D((2,2), strides=(2,2)),
        Dropout(0.5),
        Convolution2D(32, 3, 3, activation='elu', border_mode='same'),
        Convolution2D(32, 3, 3, activation='elu', border_mode='same'),
        MaxPooling2D((2,2), strides=(2,2)),
        Dropout(0.5),
        Convolution2D(64, 3, 3, activation='elu', border_mode='same'),
        Convolution2D(64, 3, 3, activation='elu', border_mode='same'),
        MaxPooling2D((2,2), strides=(2,2)),
        Dropout(0.5),
        Convolution2D(128, 3, 3, activation='elu', border_mode='same'),
        Convolution2D(128, 3, 3, activation='elu', border_mode='same'),
        UpSampling2D(size=(2,2)),
        Dropout(0.5),
        Convolution2D(64, 3, 3, activation='elu', border_mode='same'),
        Convolution2D(64, 3, 3, activation='elu', border_mode='same'),
        UpSampling2D(size=(2,2)),
        Dropout(0.5),
        Convolution2D(32, 3, 3, activation='elu', border_mode='same'),
        Convolution2D(32, 3, 3, activation='elu', border_mode='same'),
        UpSampling2D(size=(2,2)),
        Dropout(0.5),
        Convolution2D(16, 3, 3, activation='elu', border_mode='same'),
        Convolution2D(16, 3, 3, activation='elu', border_mode='same'),
        UpSampling2D(size=(2,2)),
        Dropout(0.5),
        Convolution2D(8, 3, 3, activation='elu', border_mode='same'),
        Convolution2D(8, 3, 3, activation='elu', border_mode='same'),
        Convolution2D(1, 1, 1, activation='sigmoid')
    ])

def unet(input_shape):
    i = Input(input_shape)
    Lambda(lambda x: x / 255.0 - 0.5)
    c0 = Convolution2D(3, 1, 1, init='glorot_normal')(i)
    c1 = Convolution2D(8, 3, 3, activation='elu', border_mode='same')(c0)
    c1 = Convolution2D(8, 3, 3, activation='elu', border_mode='same')(c1)
    m1 = MaxPooling2D(pool_size=(2,2))(c1)
    d1 = Dropout(0.5)(m1)
    c2 = Convolution2D(16, 3, 3, activation='elu', border_mode='same')(d1)
    c2 = Convolution2D(16, 3, 3, activation='elu', border_mode='same')(c2)
    m2 = MaxPooling2D(pool_size=(2,2))(c2)
    d2 = Dropout(0.5)(m2)
    c3 = Convolution2D(32, 3, 3, activation='elu', border_mode='same')(d2)
    c3 = Convolution2D(32, 3, 3, activation='elu', border_mode='same')(c3)
    m3 = MaxPooling2D(pool_size=(2,2))(c3)
    d3 = Dropout(0.5)(m3)
    c4 = Convolution2D(64, 3, 3, activation='elu', border_mode='same')(d3)
    c4 = Convolution2D(64, 3, 3, activation='elu', border_mode='same')(c4)
    m4 = MaxPooling2D(pool_size=(2,2))(c4)
    d4 = Dropout(0.5)(m4)
    c5 = Convolution2D(128, 3, 3, activation='elu', border_mode='same')(d4)
    c5 = Convolution2D(128, 3, 3, activation='elu', border_mode='same')(c5)
    u1 = UpSampling2D(size=(2,2))(c5)
    d5 = Dropout(0.5)(u1)
    c6 = merge([d5,c4], mode='concat', concat_axis=3)
    c6 = Convolution2D(64, 3, 3, activation='elu', border_mode='same')(c6)
    c6 = Convolution2D(64, 3, 3, activation='elu', border_mode='same')(c6)
    u2 = UpSampling2D(size=(2,2))(c6)
    d6 = Dropout(0.5)(u2)
    c7 = merge([d6,c3], mode='concat', concat_axis=3)
    c7 = Convolution2D(32, 3, 3, activation='elu', border_mode='same')(c7)
    c7 = Convolution2D(32, 3, 3, activation='elu', border_mode='same')(c7)
    u3 = UpSampling2D(size=(2,2))(c7)
    d7 = Dropout(0.5)(u3)
    c8 = merge([d7,c2], mode='concat', concat_axis=3)
    c8 = Convolution2D(16, 3, 3, activation='elu', border_mode='same')(c8)
    c8 = Convolution2D(16, 3, 3, activation='elu', border_mode='same')(c8)
    u4 = UpSampling2D(size=(2,2))(c8)
    d8 = Dropout(0.5)(u4)
    c9 = merge([d8,c1], mode='concat', concat_axis=3)
    c9 = Convolution2D(8, 3, 3, activation='elu', border_mode='same')(c9)
    c9 = Convolution2D(8, 3, 3, activation='elu', border_mode='same')(c9)
    o = Convolution2D(1, 1, 1, activation='sigmoid')(c9)
    return Model(input=i, output=o)

