import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display
from tensorflow.keras import layers

tf.config.set_soft_device_placement(True)


def plt_tracks(tracks):
    if tracks.ndim != 3:
        print('Incorrect tracks dimension')
    else:
        for i in range(len(tracks)):
            track_show(tracks[i])
        plt.colorbar().set_label('Time')
        plt.show()


def track_show(track):
    x = track[:, 0]
    y = track[:, 1]
    plt.plot(x[:], y[:], '-')
    plt.scatter(x[:], y[:], c=np.arange(len(x)), s=5)
    plt.xlabel('X')
    plt.ylabel('Y')


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    # fig = plt.figure(figsize=(4, 4))

    plt_tracks(predictions[0:1])
    plt.axis('off')

    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    plt.close()


class DCGAN:
    def __init__(self):
        self.EPOCHES = 5000
        self.noise_dim = 100
        self.num_examples_to_generate = 16
        self.BATCH_SIZE = 5
        self.BUFFER_SIZE = 5
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    @staticmethod
    def make_generator_model():
        model = tf.keras.Sequential()
        model.add(layers.Dense(37 * 256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((37, 256)))
        assert model.output_shape == (None, 37, 256)  # Note: None is the batch size

        model.add(layers.Conv1DTranspose(128, 8, strides=1, padding='same', use_bias=False))
        assert model.output_shape == (None, 37, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv1DTranspose(64, 8, strides=3, padding='same', use_bias=False))
        assert model.output_shape == (None, 111, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv1DTranspose(8, 8, strides=3, padding='same', use_bias=False))
        assert model.output_shape == (None, 333, 8)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv1DTranspose(2, 8, strides=3, padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 999, 2)

        return model

    @staticmethod
    def make_discriminator_model():
        model = tf.keras.Sequential()
        model.add(layers.Conv1D(64, 3, dilation_rate=2, padding='same',
                                input_shape=[999, 2]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv1D(128, 3, dilation_rate=4, padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())
        model.add(layers.GaussianNoise(0.1))
        model.add(layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self, dataset, epochs):
        start = time.time()
        g_loss = []
        d_loss = []
        for epoch in range(epochs):
            for image_batch in dataset:
                gen_loss, disc_loss = self.train_step(image_batch)

                # Produce images for the GIF as you go
                # display.clear_output(wait=True)
                # generate_and_save_images(generator,
                #                        epoch + 1,
                #                         seed)

                # Save the model every 15 epochs
                # if (epoch + 1) % 15 == 0:
                #  checkpoint.save(file_prefix = checkpoint_prefix)
                # if (epoch + 1) % 50 == 0:
                #     display.clear_output(wait=True)
                #     generate_and_save_images(self.generator, epochs,
                #                              tf.random.normal([self.num_examples_to_generate, self.noise_dim]))
                #     print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                # # print(gen_loss, disc_loss)
                g_loss.append(gen_loss), d_loss.append(disc_loss)

        # Generate after the final epoch
        # display.clear_output(wait=True)
        generate_and_save_images(self.generator, epochs,
                                 tf.random.normal([self.num_examples_to_generate, self.noise_dim]))
        return g_loss, d_loss


if __name__ == '__main__':
    dcgan = DCGAN()
