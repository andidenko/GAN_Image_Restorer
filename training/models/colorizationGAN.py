import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Model
from net_elements import downsample
from utils.image_processing_utils import generate_real_samples

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 2], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, apply_batchnorm=False)(x)
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

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def Generator():
    base_model = sm.Unet(backbone_name='resnet50',
                         encoder_weights='imagenet',
                         input_shape=(256, 256, 3),
                         classes=2,
                         activation='tanh',
                         encoder_freeze=True,
                         decoder_block_type='transpose',
                         decoder_use_batchnorm=True)

    inp = Input(shape=(None, None, 1))
    l1 = Conv2D(3, (1, 1))(inp)
    out = base_model(l1)

    return Model(inp, out, name=base_model.name)
