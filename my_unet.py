from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, Input, \
    concatenate, Add
from keras.layers import Input, Lambda
import tensorflow as tf

def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)


def conv2d_block(
        inputs,
        use_batch_norm=True,
        dropout=0.3,
        filters=16,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same'):
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = Dropout(dropout)(c)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


def ctom_unet(
        input_shape,
        num_classes=3,
        use_batch_norm=True,
        upsample_mode='deconv',  # 'deconv' or 'simple'
        use_dropout_on_upsampling=False,
        dropout=0.3,
        dropout_change_per_layer=0.0,
        filters=16,
        num_layers=4,
        output_activation='sigmoid'):  # 'sigmoid' or 'softmax'

    if upsample_mode == 'deconv':
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    # Build U-Net model
    detail_info_32 = Input(shape=(32, 32, 3))
    detail_info_64 = Input(shape=(64, 64, 3))
    detail_info_128 = Input(shape=(128, 128, 3))
    detail_info_256 = Input(shape=(256, 256, 3))
    detail_info_512 = Input(shape=(512, 512, 3))
    detail_info_1024 = Input(shape=(1024, 1024, 3))
    luma_info_32 = Input(shape=(32, 32,3))
    luma_info_64 = Input(shape=(64, 64, 3))
    luma_info_128 = Input(shape=(128, 128, 3))
    luma_info_256 = Input(shape=(256, 256, 3))
    luma_info_512 = Input(shape=(512, 512, 3))
    detail_info = (detail_info_32, detail_info_64, detail_info_128, detail_info_256, detail_info_512, detail_info_1024)
    luma_info = (luma_info_32, luma_info_64, luma_info_128, luma_info_256, luma_info_512)

    inputs = Input(input_shape)
    x = inputs
    count = 5
    down_layers = []
    for l in range(num_layers):
        if count<=4:
            a = conv2d_block(inputs=luma_info[count], filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
            x = concatenate([x, a])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer
        count -= 1

    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
    # x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
    # x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0
    count = 0
    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding='same')(x)
        a = conv2d_block(inputs=detail_info[count], filters=1, use_batch_norm=use_batch_norm, dropout=dropout)
        x = concatenate([x, conv])
        x = Add()([x, a])

        # y = detail_info[count]
        # for i in range(0, x._keras_shape[3]-1):
        #     y = concatenate([y, detail_info[count]], axis=3)
        # x = Add()([x, y])

        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
        if count == 0:
            output_32 = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
        elif count == 1:
            output_64 = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
        elif count == 2:
            output_128 = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
        elif count == 3:
            output_256 = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
        elif count == 4:
            output_512 = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
        else:
            output_1024 = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
        count += 1

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    def extract_Y(x):
        x_y = tf.image.rgb_to_yuv(x)
        y = x_y[:,:,:,0:1]
        return y

    output = Lambda(extract_Y)(outputs)
    model = Model(inputs=[inputs, detail_info_32, detail_info_64, detail_info_128, detail_info_256, detail_info_512, detail_info_1024,
                          luma_info_32, luma_info_64, luma_info_128, luma_info_256, luma_info_512],
                  outputs=[output,outputs,output_32,output_64,output_128,output_256,output_512])
    return model

