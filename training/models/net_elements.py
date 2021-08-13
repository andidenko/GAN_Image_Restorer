import tensorflow as tf



def downsample(filters, kernel_size, strides=2, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU(0.2))
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def generator_loss(disc_generated_output, gen_output, target):
    crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = crossentropy(tf.ones_like(disc_generated_output), disc_generated_output)
    gan_loss = tf.cast(gan_loss, dtype=tf.float64)

    l1_loss = tf.reduce_mean(tf.abs((target - gen_output)))

    total_gen_loss = gan_loss + (100 * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = crossentropy(tf.ones_like(disc_real_output), disc_real_output) * 0.9
    real_loss = tf.cast(real_loss, dtype=tf.float64)

    generated_loss = crossentropy(tf.zeros_like(disc_generated_output), disc_generated_output)
    generated_loss = tf.cast(generated_loss, dtype=tf.float64)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss