import numpy as np
import os
import matplotlib.pyplot as plt


def add_noise(x, factor):
    x_noise = x + factor * np.random.normal(loc=0., scale=0.5, size=x.shape)
    x_noise = np.clip(x_noise, 0, 1)
    return x_noise


def plot_decoded_imgs(x_test, reconstructed):
    plt.figure(figsize=(20, 4))
    plt.gray()
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


def plot_error(error, legend_names, num_epochs, title):
    # fig config
    plt.figure()
    plt.grid(True)

    # plt.ylim(0, 10)
    plt.xlim(-0.5, num_epochs)
    plt.ylim(0., 1.)

    epochs = np.arange(0, num_epochs, 1)

    for i in range(len(error)):
        plt.plot(epochs, error[i][:])

    plt.title(title)
    plt.legend(legend_names, loc='upper right')

    plt.show()


if __name__ == '__main__':
    pass
