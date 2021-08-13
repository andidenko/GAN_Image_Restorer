from models.colorizationGAN import *
from utils.dataset_utils import *
from image_processing_utils import *
from skimage import color
from net_elements import generator_loss, discriminator_loss
import tensorflow as tf
import numpy as np
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
def train_step(input_image, target, iteration):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        gen_output = tf.cast(gen_output, dtype=tf.float64)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
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
        real_source, real_target = generate_real_samples(batch)
        train_step(real_source, real_target, tf.convert_to_tensor(iteration, dtype=tf.int64))
        iteration += 1
        if iteration % echo == 0:
            print (f"Iteration {iteration}, {time.time() - start_time}s")
            start_time = time.time()

            real_grayscale, real_ab = generate_real_samples(batch)
            fake_ab = generator.predict(real_grayscale)

            grayscale_images = np.zeros(shape=(BATCH_SIZE, 256, 256), dtype=np.float32)
            colorized_images = np.zeros(shape=(BATCH_SIZE, 256, 256, 3), dtype=np.float32)

            for i in range(BATCH_SIZE):
                grayscale_images[i,:,:] = color.rgb2gray(batch[i,:,:,:])

                colorized_images[i,:,:,0] = (real_grayscale[i, :, :] + 1.) * 50
                colorized_images[i,:,:,1:] = fake_ab[i, :, :, :] * 128
                colorized_images[i,:,:,:] = color.lab2rgb(colorized_images)


            vizualize_perfomance(batch, grayscale_images, colorized_images)
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

    generator.save("/resnet50GAN/generator")
    discriminator.save("/resnet50GAN/discriminator")
