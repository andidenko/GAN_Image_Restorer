import numpy as np
from skimage import color
import cv2
import matplotlib.pyplot as plt

def resize_images(images):
    lr_images = list()
    for i in range(images.shape[0]):
        lr_image = cv2.resize(images[i,:,:,:], (images.shape[1]//2, images.shape[2]//2))
        lr_images.append(lr_image)
    lr_images = np.asarray(lr_images)
    return lr_images


def add_noise(batch):
	samples, row, col, ch = batch.shape
	mean = 0
	var = 10
	noise = np.random.normal(mean,var,(samples, row, col, ch))
	noise = noise.reshape(samples, row, col, ch)
	noisy_batch = batch + noise
	noisy_batch = np.clip(noisy_batch, 0, 255)
	return noisy_batch


def generate_real_samples(batch):
    X = list()
    y = list()
    for i in range(batch.shape[0]):
        lab_image = color.rgb2lab(batch[i])
        lab_image[:,:,0] = lab_image[:,:,0] / 50. - 1.
        lab_image[:,:,1:] = lab_image[:,:,1:] / 128.
        X.append(lab_image[:,:,0])
        y.append(lab_image[:,:,1:])
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


def vizualize_perfomance(true_images, corrupted_images, generated_images):
	n_samples = true_images.shape[0]
	n_cols = 3

	fig, ax = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 4 * n_samples))
	if ax.ndim == 1:
        ax = np.array([ax])
    for i in range(n_samples):
    	ax[i, 0].imshow(true_images[i, :, :, :])
    	ax[i, 1].imshow(corrupted_images[i, :, :, :])
    	ax[i, 2].imshow(generated_images[i, :, :, :])

    for i in range(n_samples):
        for j in range(n_cols):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    labels = ['true', 'corrupted', 'generated']
    for i in range(n_cols):
        ax[n_samples - 1, i].set_xlabel(labels[i], fontsize=14)

    fig.subplots_adjust(wspace=0, hspace=0.05)
    fig.savefig('results.png', dpi=fig.dpi)
    plt.close()

