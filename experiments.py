from keras import metrics
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import Utils
import numpy as np
from autoenconder import AutoEncoder, ErrorsCallback
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from sklearn import metrics
import time
from keras import backend as K


def ex_3_2(num_of_nodes_in_hidden_layers):
    x_train = np.loadtxt('binMNIST_data/bindigit_trn.csv', delimiter=',', dtype=float)
    x_test = np.loadtxt('binMNIST_data/bindigit_tst.csv', delimiter=',', dtype=float)

    y_train = np.loadtxt('binMNIST_data/targetdigit_trn.csv', delimiter=',', dtype=float)
    y_test = np.loadtxt('binMNIST_data/targetdigit_tst.csv', delimiter=',', dtype=float)

    target_train = to_categorical(y_train, num_classes=10)

    n_epochs = 10
    batch_size = 128

    output_train = x_train

    final_model = Sequential()

    for hidden_dim in num_of_nodes_in_hidden_layers:
        model1 = AutoEncoder(encode_dim=hidden_dim, input_dim=output_train.shape[1], l2_value=0.,
                             encode_activation='relu')

        # error_callback = ErrorsCallback(output_train, output_train, output_test, output_test)

        model1.train(x_train=output_train, n_epochs=n_epochs, batch_size=batch_size, callbacks=[],
                     loss='binary_crossentropy')

        final_model.add(model1.model.layers[-2])

        # output_train = model1.encoder.predict(output_train)

        get_hidden_layer_output = K.function([model1.model.layers[0].input],
                                          [model1.model.layers[-2].output])

        output_train = get_hidden_layer_output([output_train])[0]

        print(output_train.shape)

    final_model.add(Dense(10, activation='softmax'))

    final_model.compile(optimizer=SGD(lr=0.1, momentum=0.9), loss='mean_squared_error', metrics=['accuracy'])

    history = final_model.fit(x_train, target_train,
                              epochs=n_epochs,
                              batch_size=batch_size,
                              shuffle=True)

    predictions = final_model.predict(x_train)
    predictions_test = final_model.predict(x_test)

    # print(predictions)

    predictions = np.argmax(final_model.predict(x_test), axis=1)
    print(metrics.classification_report(y_test, predictions))


def ex_3_1_v2():
    x_train = np.loadtxt('binMNIST_data/bindigit_trn.csv', delimiter=',', dtype=float)
    x_test = np.loadtxt('binMNIST_data/bindigit_tst.csv', delimiter=',', dtype=float)

    first_dot = False
    second_dot = False
    third_dot = False
    fourth_dot = True
    error_trains = []
    n_epochs = 200

    if fourth_dot:
        n_nodes = [100, 200, 400]
        for nodes in n_nodes:
            error_callback = ErrorsCallback(x_train, x_train, x_test, x_test)
            model1 = AutoEncoder(encode_dim=nodes, input_dim=784, l2_value=0., encode_activation='relu')

            model1.train(x_train=x_train, n_epochs=n_epochs, batch_size=128, callbacks=[error_callback],
                         loss='binary_crossentropy')
            # reconstructed = model1.predict(x_test)

            w = model1.model.layers[-1].get_weights()[0]

            for unit in w:
                plt.imshow(unit.reshape(28, 28))
                plt.show()
            error_trains.append(error_callback.mse_train)

    if third_dot:
        noise_factors = [0., 0.1, 0.2, 0.3, 0.4]
        for noise in noise_factors:
            x_train_noisy = Utils.add_noise(x_train, noise)
            x_test_noisy = Utils.add_noise(x_test, noise)

            error_callback = ErrorsCallback(x_train_noisy, x_train, x_test_noisy, x_test)

            model1 = AutoEncoder(encode_dim=128, input_dim=784, l2_value=0., encode_activation='relu')

            model1.train(x_train=x_train, n_epochs=n_epochs, batch_size=128, callbacks=[error_callback],
                         loss='binary_crossentropy')

            reconstructed = model1.predict(x_test)

            error_trains.append(error_callback.mse_train)

        legends = ['noise: 0%', 'noise: 10%', 'noise: 20%', 'noise: 30%', 'noise: 40%']
        Utils.plot_error(error_trains, legend_names=legends, num_epochs=n_epochs, title='Error on Epochs with Noise')

    if second_dot:
        activations = ['sigmoid', 'relu']
        error_trains = []
        n_epochs = 200

        for activation in activations:
            error_callback = ErrorsCallback(x_train, x_train, x_test, x_test)

            model1 = AutoEncoder(encode_dim=900, input_dim=784, l2_value=0.01, encode_activation=activation)

            model1.train(x_train=x_train, n_epochs=n_epochs, batch_size=128, callbacks=[error_callback],
                         loss='binary_crossentropy')

            reconstructed = model1.predict(x_test)

            error_trains.append(error_callback.mse_train)

            # Utils.plot_decoded_imgs(x_test, reconstructed)

        legends = ['sigmoid', 'relu']
        Utils.plot_error(error_trains, legend_names=legends, num_epochs=n_epochs,
                         title='1500 Nodes l2=0.0001')

    if first_dot:
        encod_dims = [512, 256, 128, 64, 32]
        error_trains = []
        n_epochs = 100
        times = []
        for encode_dim in encod_dims:
            start_time = time.time()
            error_callback = ErrorsCallback(x_train, x_train, x_test, x_test)

            model1 = AutoEncoder(encode_dim=encode_dim, input_dim=784)

            model1.train(x_train=x_train, n_epochs=n_epochs, batch_size=128, callbacks=[error_callback],
                         loss='binary_crossentropy')

            reconstructed = model1.predict(x_test)
            end_time = time.time() - start_time
            times.append(str(round(end_time, 2)))
            error_trains.append(error_callback.mse_train)

            # Utils.plot_decoded_imgs(x_test, reconstructed)

        legends = ['512 Nodes, ' + times[0] + ' sec',
                   '256 Nodes, ' + times[1] + ' sec',
                   '128 Nodes, ' + times[2] + ' sec',
                   '64 Nodes, ' + times[3] + ' sec',
                   '32 Nodes, ' + times[4] + ' sec']
        Utils.plot_error(error_trains, legend_names=legends, num_epochs=len(error_trains[0]),
                         title='Auto encoder for various encoding dimensions')


if __name__ == "__main__":
    # x_train = np.loadtxt('binMNIST_data/bindigit_trn.csv', delimiter=',', dtype=float)
    # x_test = np.loadtxt('binMNIST_data/bindigit_tst.csv', delimiter=',', dtype=float)

    # x_noise_train = Utils.add_noise(x_train, 0.3)
    # x_noise_test = Utils.add_noise(x_test, 0.3)

    # error_callback = ErrorsCallback(x_noise_train, x_train, x_noise_test, x_test)
    #
    # model1 = AutoEncoder(encode_dim=128, input_dim=784, l2_value=0., encode_activation='relu')
    #
    # model1.train(x_train=x_train, n_epochs=200, batch_size=64, callbacks=[error_callback],
    #              loss='binary_crossentropy')
    #
    # reconstructed = model1.predict(x_test)
    #
    # plt.gca()
    # plt.imshow(x_test[10].reshape(28, 28))
    # plt.show()
    #
    # plt.imshow(x_noise_test[10].reshape(28, 28))
    # plt.show()
    #
    # plt.imshow(reconstructed[10].reshape(28, 28))
    # plt.show()

    # ex_3_1_v2()
    ex_3_2([512, 256, 128])
