from keras import metrics
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import Utils
import numpy as np
from autoenconder import AutoEncoder, ErrorsCallback
from keras.models import Model
from keras.layers import Input, Dense
from sklearn import metrics

encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
input_dimenions = 784
batch_size = 32
epochs = 30


def ex_3_1():
    # x_train, _, x_test, _ = Utils.load_mnist()
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    x_train = np.loadtxt('binMNIST_data/bindigit_trn.csv', delimiter=',', dtype=float)
    x_test = np.loadtxt('binMNIST_data/bindigit_tst.csv', delimiter=',', dtype=float)

    # this is our input placeholder
    input_img = Input(shape=(input_dimenions,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_dimenions, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)

    # autoencoder = AutoEncoder(input_dim=x_train.shape[1], encode_dim=32)

    encoder = Model(input_img, encoded)

    encoded_input = Input(shape=(32,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True)
    # validation_data=(x_test, x_test))

    # autoencoder.train(x_train, x_test, n_epochs=15)

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    Utils.plot_decoded_imgs(x_test, decoded_imgs)


def ex_3_2(num_of_nodes_in_hidden_layers):
    x_train = np.loadtxt('binMNIST_data/bindigit_trn.csv', delimiter=',', dtype=float)
    x_test = np.loadtxt('binMNIST_data/bindigit_tst.csv', delimiter=',', dtype=float)

    y_train = np.loadtxt('binMNIST_data/targetdigit_trn.csv', delimiter=',', dtype=float)
    y_test = np.loadtxt('binMNIST_data/targetdigit_tst.csv', delimiter=',', dtype=float)

    target_train = to_categorical(y_train, num_classes=10)

    input_layer = Input(shape=(input_dimenions,))
    layer = input_layer
    # encode
    for hidden_dim in num_of_nodes_in_hidden_layers:
        layer = Dense(hidden_dim, activation='relu')(layer)
    num_of_nodes_in_hidden_layers.pop(-1)

    # decode
    for hidden_dim in reversed(num_of_nodes_in_hidden_layers):
        layer = Dense(hidden_dim, activation='relu')(layer)
    layer = Dense(input_dimenions, activation='relu')(layer)

    out = Dense(10, activation='sigmoid')(layer)

    model = Model(input_layer, out)
    model.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x_train, target_train,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True)

    predictions = np.argmax(model.predict(x_test), axis=1)
    print(metrics.classification_report(y_test, predictions))


def ex_3_1_v2():
    x_train = np.loadtxt('binMNIST_data/bindigit_trn.csv', delimiter=',', dtype=float)
    x_test = np.loadtxt('binMNIST_data/bindigit_tst.csv', delimiter=',', dtype=float)

    error_callback = ErrorsCallback(x_train, x_train, x_test, x_test)

    model = AutoEncoder(encode_dim=32, input_dim=784)
    model.train(x_train=x_train, n_epochs=100, batch_size=64, callbacks=[error_callback])
    reconstructed = model.predict(x_test)


    plt.plot(error_callback.mse_test)
    plt.show()
    Utils.plot_decoded_imgs(x_test, reconstructed)


if __name__ == "__main__":
    # ex_3_1()
    ex_3_1_v2()
    # ex_3_2([128, 64, 32])
