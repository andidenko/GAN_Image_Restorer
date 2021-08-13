from models.SRGAN import *
from image_processing_utils import *
from net_elements import discriminator_loss
from utils.dataset_utils import *
import tensorflow as tf
import numpy as np
import cv2
import time
import datetime

BATCH_SIZE = 16
ITERATIONS = 125000
ECHO = 1000
SHUFFLE = 1000

generator_optimizer = tf.keras.optimizers.Adam(2e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)

log_dir="logs/"
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(lr_images, hr_images, iteration):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        sr_images = generator(lr_images, training=True)

        sr_images = tf.cast(sr_images, dtype=tf.float64)

        disc_real_output = discriminator(hr_images, training=True)
        disc_generated_output = discriminator(sr_images, training=True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, sr_images, hr_images)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    if iteration % 100 == 0:
        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=iteration)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=iteration)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=iteration)
            tf.summary.scalar('disc_loss', disc_loss, step=iteration)


def fit(train_dataset, iterations, echo):
    start_time = time.time()
    iteration = 0
    for batch in train_dataset:
        hr_images = batch.numpy().astype(np.float64)
        lr_images = resize_images(hr_images)

        hr_images = (hr_images - 127.5) / 127.5
        lr_images = (lr_images - 127.5) / 127.5

        train_step(lr_images, hr_images, tf.convert_to_tensor(iteration, dtype=tf.int64))

        iteration += 1
        if iteration % echo == 0:
            print (f"Iteration {iteration}, {time.time() - start_time}s")
            start_time = time.time()

            sr_images = generator(lr_images)

            hr_images = (hr_images + 1) * 0.5
            lr_images = (lr_images + 1) * 0.5
            sr_images = (sr_images + 1) * 0.5

            resized_images = np.zeros(shape=(BATCH_SIZE, 256, 256, 3))

            for i in range(BATCH_SIZE):
                resized_images[i,:,:,:] = cv2.resize(lr_images[i,:,:,:], (256, 256), interpolation=cv2.INTER_NEAREST)

            vizualize_perfomance(hr_images, resized_images, sr_images)
            manager.save()
        if iteration >= iterations:
            break


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

    train_dataset = get_dataset("/tf_dataset/train*", SHUFFLE, BATCH_SIZE)

    generator = Generator()
    discriminator = Discriminator()

    checkpoint_dir = '/checkpoints'
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

    fit(train_dataset, ITERATIONS, ECHO)

    generator.save("/SRGAN/generator")
    discriminator.save("/SRGAN/discriminator")