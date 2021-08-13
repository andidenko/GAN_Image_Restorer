import tensorflow as tf
from image_processing import *


class GAN():
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def process(self, input_image):
        raise NotImplementedError()


class ColorizationGAN(GAN):
    def __init__(self, model_path):
        super().__init__(model_path)

    def process(self, input_image):
        if input_image.shape[2] != 1:
            raise ValueError("Image shape should be (h, w, 1)")

        input_image = input_image / 50. - 1.
        input_tensor = input_image[np.newaxis, ...]
        ab = self.model.predict(input_tensor)

        output_image = lab_output2image(input_image, ab)
        return output_image


class DenoisingGAN(GAN):
    def __init__(self, model_path):
        super().__init__(model_path)

    def process(self, input_image):
        input_tensor = image2tensor(input_image)
        output_tensor = self.model.predict(input_tensor)
        output_image = output2image(output_tensor)
        return output_image


class SRGAN(GAN):
    def __init__(self, model_path):
        super().__init__(model_path)

    def process(self, input_image):
        input_tensor = image2tensor(input_image)
        output_tensor = self.model.predict(input_tensor)
        output_image = output2image(output_tensor)
        return output_image

