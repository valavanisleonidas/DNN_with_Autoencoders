import Utils
import numpy as np
from autoenconder import AutoEncoder
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense


def plot_decoded_imgs(x_test, reconstructed):
    plt.figure(figsize=(20, 4))
    for i in range(10):
        # display original
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, 10, i + 11)
        plt.imshow(reconstructed[i].reshape(28, 28))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


if __name__ == "__main__":
    x_train, _, x_test, _ = Utils.load_mnist()
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_img = Input(shape=(784,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(784, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)

    # autoencoder = AutoEncoder(input_dim=x_train.shape[1], encode_dim=32)

    encoder = Model(input_img, encoded)

    encoded_input = Input(shape=(32,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    
    # autoencoder.train(x_train, x_test, n_epochs=15)

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    plot_decoded_imgs(x_test, decoded_imgs)
