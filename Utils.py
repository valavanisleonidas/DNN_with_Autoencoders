import numpy as np
import os




def _load_filename(prefix, path):
    int_type = np.dtype('int32').newbyteorder('>')
    n_meta_data_bytes = 4 * int_type.itemsize

    data = np.fromfile(path + "/" + prefix + '-images-idx3-ubyte', dtype='ubyte')

    magic_bytes, n_images, width, height = np.frombuffer(data[:n_meta_data_bytes].tobytes(), int_type)
    data = data[n_meta_data_bytes:].astype(dtype='float32').reshape([n_images, width, height])

    labels = np.fromfile(path + "/" + prefix + '-labels-idx1-ubyte',
                         dtype='ubyte')[2 * int_type.itemsize:]

    return data, labels


def load_mnist(path=None):
    print("Loading mnist....")
    """
        Load mnist

        link ~ http://yann.lecun.com/exdb/mnist/

        Data Description:

        The data is stored in a very simple file format designed for storing vectors and multidimensional matrices. 
        General info on this format is given at the end of this page, but you don't need to read that to use the data 
        files. All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel 
        processors. Users of Intel processors and other low-endian machines must flip the bytes of the header. 

        There are 4 files:

        train-images-idx3-ubyte: training set images
        train-labels-idx1-ubyte: training set labels
        t10k-images-idx3-ubyte:  test set images
        t10k-labels-idx1-ubyte:  test set labels

    """
    if path is None:
        root = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
        path = os.path.join(root, 'data/mnist/')

    training_images, training_labels = _load_filename("train", path)
    test_images, test_labels = _load_filename("t10k", path)

    # Make the images Binary
    training_images = np.where(training_images > 128, 1, 0)
    test_images = np.where(test_images > 128, 1, 0)
    print("-----------------------------------------")
    print("           mnist is Loaded")
    print("-----------------------------------------")
    return training_images, training_labels, test_images, test_labels


if __name__ == '__main__':
    load_mnist()
