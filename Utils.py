import numpy as np
import os
import matplotlib.pyplot as plt
import copy


def add_noise(x, percentage):
    noisy = copy.deepcopy(x)

    for i, img in enumerate(copy.deepcopy(x)):
        N = len(img)
        n_indices = int(percentage * N)
        indices = np.random.choice(np.arange(N), n_indices, replace=False)
        img[indices] += np.random.normal(loc=0., scale=1, size=img[indices].shape)
        img = np.clip(img, 0, 1)
        noisy[i] = img

    return noisy


def plot_decoded_imgs(x_test, reconstructed, title = ''):
    fig = plt.figure(figsize=(20, 4))
    plt.gray()
    counter = 0
    for i in [156, 155, 259, 1584, 154, 160, 383, 388, 253, 153]:
        # display original
        ax = plt.subplot(2, 10, counter + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, 10, counter + 11)
        plt.imshow(reconstructed[i].reshape(28, 28))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        counter += 1

    fig.suptitle(title)
    plt.show()


def show_images(images, rows, columns, title=None, dimensions=(28, 28)):
    ncols = int(columns + 1)
    nrows = int(rows + 1)

    # create the plots
    fig = plt.figure()
    axes = [fig.add_subplot(nrows, ncols, r * ncols + c) for r in range(1, nrows) for c in range(1, ncols)]

    counter = 0
    # add some data
    for ax in axes:
        if counter >= len(images):
            ax.axis('off')
            continue
        image = images[counter]
        ax.imshow(image.reshape(dimensions[0], dimensions[1]))
        counter += 1

    # remove the x and y ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    plt.show()


def plot_error(error, legend_names, num_epochs, title):
    # fig config
    plt.figure()
    plt.grid(True)

    # plt.ylim(0, 10)
    plt.xlim(0., num_epochs)
    plt.ylim(0.5, 1.)

    epochs = np.arange(0, num_epochs, 1)

    for i in range(len(error)):
        plt.plot(epochs, error[i][:])

    plt.xlabel('Epochs')
    plt.ylabel('Error')

    plt.title(title)
    plt.legend(legend_names, loc='upper right')

    plt.show()


def plot_acuracy(accuracy, legend_names, num_epochs, title):
    # fig config
    plt.figure()
    plt.grid(True)

    # plt.ylim(0, 10)
    plt.xlim(0., num_epochs)
    plt.ylim(0.5, 1.)

    epochs = np.arange(0, num_epochs, 1)

    for i in range(len(accuracy)):
        plt.plot(epochs, accuracy[i][:])

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.title(title)
    plt.legend(legend_names, loc='lower right')

    plt.show()


def plot_sparseness(sparseness, legend_names, nodes, title):
    # fig config
    plt.figure()
    plt.grid(True)

    # plt.ylim(0, 10)
    plt.xlim(-0.5, nodes[-1])
    plt.ylim(0., 1.)

    plt.xlabel('Number of nodes in hidden layer')
    plt.ylabel('% of sparseness')

    for i in range(len(sparseness)):
        plt.plot(nodes, sparseness[i][:])

    plt.title(title)
    plt.legend(legend_names, loc='upper right')

    plt.show()


if __name__ == '__main__':
    pass
