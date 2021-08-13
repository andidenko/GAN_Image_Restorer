import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from net_elements import downsample, upsample
from utils.image_processing_utils import resize_images


def res_block_gen(model, kernal_size, filters, strides):
    gen = model

    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = LeakyReLU(0.2)(model)

    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)

    model = add([gen, model])

    return model


def up_sampling_block(model, kernal_size, filters, strides):
    model = Conv2DTranspose(filters, kernal_size, strides=strides,
                            padding='same',
                            use_bias=False)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization()(model)

    return model


def Generator():
    gen_input = Input(shape=[128, 128, 3])

    model = Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(gen_input)
    model = LeakyReLU(0.2)(model)

    gen_model = model

    for index in range(10):
        model = res_block_gen(model, 3, 64, 1)

    model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = add([gen_model, model])

    for index in range(1):
        model = up_sampling_block(model, 3, 128, 2)

    model = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(model)
    model = Activation('tanh')(model)

    generator_model = Model(inputs=gen_input, outputs=model)
    return generator_model


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')

    down1 = downsample(64, 4, apply_batchnorm=False)(inp)  
    down2 = downsample(128, 4)(down1)  
    down3 = downsample(256, 4)(down2)  

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) 

    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2) 
    last = tf.keras.layers.LeakyReLU()(last)

    return tf.keras.Model(inputs=inp, outputs=last)


vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
vgg19.trainable = False
for l in vgg19.layers:
    l.trainable = False
vgg = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
vgg.trainable = False


def vgg_loss(y_true, y_pred):
    return K.mean(K.square(vgg(y_true) - vgg(y_pred)))


def generator_loss(disc_generated_output, gen_output, target):
    crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = crossentropy(tf.ones_like(disc_generated_output), disc_generated_output)
    gan_loss = tf.cast(gan_loss, dtype=tf.float64)

    content_loss = vgg_loss(target, gen_output)
    content_loss = tf.cast(content_loss, dtype=tf.float64)

    total_gen_loss = (1e-3 * gan_loss) + content_loss

    return total_gen_loss, gan_loss, content_loss