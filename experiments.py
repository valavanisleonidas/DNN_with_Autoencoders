from keras import metrics
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import Utils
import numpy as np
from autoenconder import AutoEncoder, ErrorsCallback
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Dense, BatchNormalization, Dropout
import time
from keras import backend as K

from keras.utils import plot_model


def ex_3_2(num_of_nodes_in_hidden_layers):
    x_train = np.loadtxt('binMNIST_data/bindigit_trn.csv', delimiter=',', dtype=float)
    x_test = np.loadtxt('binMNIST_data/bindigit_tst.csv', delimiter=',', dtype=float)

    y_train = np.loadtxt('binMNIST_data/targetdigit_trn.csv', delimiter=',', dtype=float)
    y_test = np.loadtxt('binMNIST_data/targetdigit_tst.csv', delimiter=',', dtype=float)

    target_train = to_categorical(y_train, num_classes=10)
    target_test = to_categorical(y_test, num_classes=10)

    n_epochs = 10
    batch_size = 128
    save = False
    load = False

    if load is True:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        final_model = model_from_json(loaded_model_json)
        final_model.load_weights("model.h5")
        print("Loaded model from disk")

    else:
        output_train = x_train

        final_model = Sequential()

        for hidden_dim in num_of_nodes_in_hidden_layers:
            model1 = AutoEncoder(encode_dim=hidden_dim, input_dim=output_train.shape[1], l2_value=0.,
                                 encode_activation='relu', add_batch_norm=True)

            model1.train(x_train=output_train, n_epochs=n_epochs, batch_size=batch_size, callbacks=[],
                         loss='mean_squared_error')

            final_model.add(model1.model.layers[-3])
            final_model.add(model1.model.layers[-2])

            get_hidden_layer_output = K.function([model1.model.layers[0].input],
                                                 [model1.model.layers[-2].output])
            output_train = get_hidden_layer_output([output_train])[0]

            print(output_train.shape)

        final_model.add(Dense(10, activation='sigmoid'))

        error_callback = ErrorsCallback(x_train, target_train, x_test, target_test)

        final_model.compile(optimizer=SGD(lr=.1, decay=0.001, momentum=.85, nesterov=True), loss='mean_squared_error',
                            metrics=['accuracy'])

        history = final_model.fit(x_train, target_train,
                                  epochs=n_epochs,
                                  batch_size=batch_size,
                                  callbacks=[],
                                  validation_data=(x_test, target_test),
                                  shuffle=True)

        if save is True:
            model_json = final_model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            final_model.save_weights("model.h5")
            print("Saved model to disk")

    # plot_model(final_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    final_model.compile(optimizer=SGD(lr=.1, decay=0.001, momentum=.85, nesterov=True), loss='mean_squared_error',
                        metrics=['accuracy'])

    test_preds = final_model.predict(x_test)

    predictions = np.argmax(test_preds, axis=1)
    true_digits = np.argmax(target_test, axis=1)
    n_correct = np.sum(np.equal(predictions, true_digits).astype(int))
    total = float(len(predictions))
    print("Test Accuracy:", round(n_correct / total, 3) * 100, "%")

    errors = [history.history['loss'], history.history['val_loss']]
    legends = ['mse_train', 'mse_test']
    Utils.plot_error(errors, legend_names=legends, num_epochs=n_epochs, title="Performance")


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
    # ex_3_1_v2()
    ex_3_2([512, 256, 128])

    # ex_3_2_expeirment()
