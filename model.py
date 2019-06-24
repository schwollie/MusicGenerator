import pickle

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import BatchNormalization, Activation
from keras.layers import Input, Dense, Reshape, Conv2D, Flatten, InputLayer, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD


class AutoEncoder:
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 2000
        self.img_shape = (self.img_rows, self.img_cols, 1)

        sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        self.encoder = self.build_encoder()
        self.encoder.compile(optimizer=sgd, loss='binary_crossentropy')
        self.decoder = self.build_decoder()
        self.decoder.compile(optimizer=sgd, loss='binary_crossentropy')

        z = Input(shape=self.img_shape)
        encoded = self.encoder(z)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(z, decoded)
        self.autoencoder.compile(loss='binary_crossentropy', optimizer=sgd)

        self.load()

    def build_encoder(self):
        model = Sequential()
        model.add(Conv2D(input_shape=self.img_shape, filters=5, kernel_size=3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=10, kernel_size=4, padding="same", strides=(2, 1), use_bias=False))
        model.add(Conv2D(filters=1, kernel_size=3, padding="same", use_bias=False))
        model.add(Activation(activation="tanh"))

        model.summary()

        return model

    def build_decoder(self):
        model = Sequential()
        # model.add(Conv2D(input_shape=(self.img_rows/2, self.img_cols/2, 1), filters=5, kernel_size=3, padding="same"))
        model.add(Conv2DTranspose(input_shape=(self.img_rows / 2, self.img_cols, 1),
                                  filters=15, kernel_size=4, padding="same", strides=(2, 1), use_bias=False))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=1, kernel_size=3, padding="same", use_bias=False))
        model.add(Activation(activation="tanh"))

        model.summary()

        return model

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, x):
        return self.decoder.predict(x)

    def train(self, features, epochs=1, batch_size=5):
        # nfeatures = features[0:10]
        # tfeatures = features[10:20]

        for epoch in range(epochs):
            self.autoencoder.fit(features, features, batch_size=batch_size)
            print(epoch + 1)

        self.autoencoder.evaluate(features, features)
        self.save()

    def save(self):
        self.encoder.save_weights("data/model-encoder.file", overwrite=True)
        self.decoder.save_weights("data/model-decoder.file", overwrite=True)

    def load(self):
        try:
            self.encoder.load_weights("data/model-encoder.file")
            self.decoder.load_weights("data/model-decoder.file")
            z = Input(shape=self.img_shape)
            encoded = self.encoder(z)
            decoded = self.decoder(encoded)
            self.autoencoder = Model(z, decoded)
            self.autoencoder.compile(loss='binary_crossentropy', optimizer="adadelta")
        except FileNotFoundError:
            print("continue without loading")
        except OSError:
            print("continue without loading")


class GAN:
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 2000
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

        model.add(InputLayer(input_shape=noise_shape))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(2048))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(4000))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Reshape((16, 250, 1)))
        model.add(Conv2DTranspose(input_shape=(16, 250, 1), filters=10, kernel_size=4, padding="same", strides=2))
        # model.add(UpSampling2D(2))
        model.add(Conv2DTranspose(filters=10, kernel_size=4, padding="same", strides=2))
        # model.add(UpSampling2D(2))
        model.add(Conv2DTranspose(filters=10, kernel_size=3, padding="same", strides=2))
        # model.add(UpSampling2D(2))
        model.add(Conv2D(filters=10, kernel_size=4, padding="same"))
        model.add(Conv2D(filters=1, kernel_size=3, padding="same", activation="tanh"))
        # model.add(Dense(np.prod((self.img_shape[0]/2, self.img_shape[1]/2, self.img_shape[2]/2)), activation='tanh'))

        model.summary()

        # noise = Input(shape=noise_shape)
        # img = model(noise)

        return model

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(InputLayer(img_shape))
        model.add(Conv2D(padding="valid", filters=10, kernel_size=10, data_format="channels_last"))
        model.add(Conv2D(padding="valid", filters=32, kernel_size=10, strides=2))
        # model.add(MaxPool2D(pool_size=(2, 2), padding="valid"))   use strides instead of pool!
        model.add(Conv2D(padding="valid", filters=32, kernel_size=10, strides=2))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(MaxPool2D(strides=2, pool_size=(2, 2), padding="valid"))
        model.add(Conv2D(padding="valid", filters=32, kernel_size=10))
        model.add(Conv2D(padding="valid", filters=1, kernel_size=10))
        model.add(Flatten())
        model.add(Dense(800))
        model.add(Dense(400))
        model.add(Dense(100))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        # img =
        # validity = model(img)

        # return Model(img, validity)
        return model

    def train(self, features, epochs, batch_size=2, save_interval=1):  # normal batch_size is 128
        # Rescale -1 to 1

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            print("Epoch: " + str(epoch + 1) + "/" + str(epochs))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            print("training discriminator")
            # Select a random half batch of images
            idx = np.random.randint(0, features.shape[0], half_batch)
            # print(idx)
            imgs = features[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))
            print("noise: ", noise)

            # Generate a half batch of new images
            print("generator predicts noise")
            gen_imgs = self.generator.predict(noise)

            # print(half_batch)

            print("discriminator gets trained")
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            print("start training generator ...")
            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        # fig.savefig("gan/images/mnist_%d.png" % epoch)
        plt.close()

    def save(self):
        with open("data/model.file", "wb") as f:
            pickle.dump(self, f)

    def load(self, filename="data/model.file"):
        try:
            with open(filename, "rb") as f:
                gan = pickle.load(f)
                self.__dict__.update(gan)
        except FileNotFoundError:
            FileNotFoundError("couldn't load file because: ", filename, " does not exist ...")
