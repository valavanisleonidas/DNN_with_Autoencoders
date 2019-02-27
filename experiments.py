from keras.optimizers import SGD
from keras.utils import to_categorical, plot_model
import Utils
import numpy as np
from autoenconder import AutoEncoder, ErrorsCallback
from keras.models import Sequential, model_from_json
from keras.layers import Dense
import time
from keras import backend as K
from sklearn.linear_model import LogisticRegression


def ex_3_2(num_of_nodes_in_hidden_layers, n_epochs_1=10, show_hidden_weights=False, use_logistic_regression=False,
           add_decoding_layers = False):
    x_train = np.loadtxt('binMNIST_data/bindigit_trn.csv', delimiter=',', dtype=float)
    x_test = np.loadtxt('binMNIST_data/bindigit_tst.csv', delimiter=',', dtype=float)

    y_train = np.loadtxt('binMNIST_data/targetdigit_trn.csv', delimiter=',', dtype=float)
    y_test = np.loadtxt('binMNIST_data/targetdigit_tst.csv', delimiter=',', dtype=float)

    target_train = to_categorical(y_train, num_classes=10)
    target_test = to_categorical(y_test, num_classes=10)

    n_epochs = n_epochs_1
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
        output_test = x_test

        decoder_layers = []
        final_model = Sequential()
        if len(num_of_nodes_in_hidden_layers) == 0:
            final_model.add(Dense(784, input_shape=(784,)))

        counter = 0
        for hidden_dim in num_of_nodes_in_hidden_layers:
            model1 = AutoEncoder(encode_dim=hidden_dim, input_dim=output_train.shape[1], l2_value=0.,
                                 encode_activation='relu', add_batch_norm=True)

            model1.train(x_train=output_train, n_epochs=n_epochs, batch_size=batch_size, callbacks=[],
                         loss='mean_squared_error')


            # encoded
            encoder_layer = model1.model.layers[-3]
            final_model.add(encoder_layer)
            if show_hidden_weights:
                w = encoder_layer.get_weights()[0]
                w = np.array(w).T
                print(np.array(w.shape))
                rows_cols = [[20, 20], [20, 20], [15, 10]]
                dimensions = [[28, 28], [23, 23], [20, 20]]

                Utils.show_images(w, rows_cols[counter][0], rows_cols[counter][1],
                                  title='Weights for each hidden unit with {0} epochs'.format(n_epochs), dimensions=dimensions[counter])
                counter += 1

            decoder_layers.append(model1.model.layers[-1])
            # batch normalization
            final_model.add(model1.model.layers[-2])

            get_hidden_layer_output = K.function([model1.model.layers[0].input],
                                                 [model1.model.layers[-2].output])
            output_train = get_hidden_layer_output([output_train])[0]
            output_test = get_hidden_layer_output([output_test])[0]

            print(output_train.shape)

        if use_logistic_regression :
            clf = LogisticRegression(random_state=0, solver='lbfgs',
                                     multi_class='multinomial').fit(output_train, y_train)
            score = clf.score(output_test,y_test)
            print("Test Accuracy using logistic with encoded data: :::", round(score, 3) * 100, "%")

            clf = LogisticRegression(random_state=0, solver='lbfgs',
                                     multi_class='multinomial').fit(x_train, y_train)
            score = clf.score(x_test, y_test)
            print("Test Accuracy using logistic from the begginging: :::", round(score, 3) * 100, "%")

        if add_decoding_layers:
            for layer in reversed(decoder_layers):
                final_model.add(layer)

            reconstructed = final_model.predict(x_test)
            Utils.plot_decoded_imgs(x_test, reconstructed, 'Reconstruction using three hidden layers')


        final_model.add(Dense(10, activation='sigmoid'))

        error_callback = ErrorsCallback(x_train, target_train, x_test, target_test)

        final_model.compile(optimizer=SGD(lr=.1, decay=0.001, momentum=.85, nesterov=True), loss='mse',
                            metrics=['accuracy'])

        history = final_model.fit(x_train, target_train,
                                  epochs=n_epochs,
                                  batch_size=batch_size,
                                  callbacks=[],
                                  validation_data=(x_test, target_test),
                                  # validation_split=0.3,
                                  shuffle=True)

        if save is True:
            model_json = final_model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            final_model.save_weights("model.h5")
            print("Saved model to disk")

    plot_model(final_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    final_model.compile(optimizer=SGD(lr=.1, decay=0.001, momentum=.85, nesterov=True), loss='mean_squared_error',
                        metrics=['accuracy'])

    test_preds = final_model.predict(x_test)

    predictions = np.argmax(test_preds, axis=1)
    true_digits = np.argmax(target_test, axis=1)
    n_correct = np.sum(np.equal(predictions, true_digits).astype(int))
    total = float(len(predictions))
    print("Test Accuracy:", round(n_correct / total, 3) * 100, "%")

    test_error = history.history['val_acc']
    errors = [history.history['loss'], history.history['val_loss']]
    legends = ['mse_train', 'mse_test']
    # Utils.plot_error(errors, legend_names=legends, num_epochs=n_epochs, title="Performance")
    return test_error


def ex_3_1():
    x_train = np.loadtxt('binMNIST_data/bindigit_trn.csv', delimiter=',', dtype=float)
    x_test = np.loadtxt('binMNIST_data/bindigit_tst.csv', delimiter=',', dtype=float)

    first_dot = False
    second_dot = False
    third_dot = True
    fourth_dot = False
    error_trains = []
    n_epochs = 50

    rows_cols = [[10, 10], [10, 20], [20, 20]]
    if fourth_dot:
        n_nodes = [100, 200, 400]
        counter = 0
        for nodes in n_nodes:
            error_callback = ErrorsCallback(x_train, x_train, x_test, x_test)
            model1 = AutoEncoder(encode_dim=nodes, input_dim=784, l2_value=0., encode_activation='relu')

            model1.train(x_train=x_train, n_epochs=n_epochs, batch_size=128, callbacks=[error_callback],
                         loss='binary_crossentropy')
            # reconstructed = model1.predict(x_test)

            w = model1.model.layers[-2].get_weights()[0]
            w = np.array(w).T

            Utils.show_images(w, rows_cols[counter][0], rows_cols[counter][1],
                              title='Weights for each hidden unit with {0} epochs'.format(n_epochs))
            counter += 1
            # for unit in w:
            #     plt.imshow(unit.reshape(28, 28))
            #     plt.show()
            error_trains.append(error_callback.mse_train)

    if third_dot:
        noise_factors = [0.1, 0.2, 0.3, 0.4]


        for noise in noise_factors:
            x_train_noisy = Utils.add_noise(x_train, noise)
            x_test_noisy = Utils.add_noise(x_test, noise)

            model1 = AutoEncoder(encode_dim=1000, input_dim=784, l2_value=0., encode_activation='relu', noise=noise)

            history = model1.train(x_train=x_train, y_train=x_train, n_epochs=n_epochs, batch_size=128, callbacks=[],
                                   loss='binary_crossentropy')

            reconstructed = model1.predict(x_test_noisy)

            error_trains.append(history.history['loss'])

            Utils.plot_decoded_imgs(x_test_noisy, reconstructed)

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
        # encod_dims = [800,850,1000, 1200, 1500]
        encod_dims = [32, 64, 128, 256, 512, 800, 850, 1000, 1200, 1500]
        error_trains = []
        n_epochs = 10
        times = []

        total_sparse = []
        regularizer = [0.1, 0.001, 0]
        for reg in regularizer:
            sparseness = []
            for encode_dim in encod_dims:
                start_time = time.time()
                error_callback = ErrorsCallback(x_train, x_train, x_test, x_test)

                model1 = AutoEncoder(encode_dim=encode_dim, input_dim=784, l2_value=reg)

                model1.train(x_train=x_train, n_epochs=n_epochs, batch_size=512, callbacks=[error_callback],
                             loss='binary_crossentropy')

                reconstructed = model1.encoder.predict(x_train)
                reconstructed = np.where(reconstructed < 0.1, 0, reconstructed)

                num = reconstructed.size
                zeros = num - np.count_nonzero(reconstructed)

                sparse = zeros / num

                sparseness.append(sparse)

                end_time = time.time() - start_time
                times.append(str(round(end_time, 2)))
                error_trains.append(error_callback.mse_train)

                # Utils.plot_decoded_imgs(x_test, reconstructed)

            total_sparse.append(sparseness)

        legends = ['% of sparseness with l2 = {0}'.format(regularizer[0]),
                   '% of sparseness with l2 = {0}'.format(regularizer[1]),
                   '% of sparseness with l2 = {0}'.format(regularizer[2])]
        Utils.plot_sparseness(total_sparse, legend_names=legends, nodes=encod_dims,
                              title='Percentage of sparseness for various encoding dimensions')


def ex_3_2_greedy():
    n_epochs = 50

    ex_3_2([512,400,300], n_epochs, add_decoding_layers=True)

    # acc = []
    # acc.append(ex_3_2([], n_epochs))
    # acc.append(ex_3_2([512], n_epochs))
    # acc.append(ex_3_2([512, 400], n_epochs))
    # acc.append(ex_3_2([512, 400, 150], n_epochs))
    # acc.append(ex_3_2([512, 400, 150], n_epochs,add_decoding_layers=True))
    #
    # legends = ['784 -> 10', '784 -> 512 -> 10', '784 -> 512 -> 400 -> 10', '784 -> 512 -> 400 -> 150 -> 10',
    #            '784 -> 512 -> 400 -> 150 -> 400 -> 512 -> 784 -> 10']
    # Utils.plot_acuracy(acc, legend_names=legends, num_epochs=n_epochs, title="Performance on deep autoencoders")


    # ex_3_2([529, 400, 144], n_epochs, show_hidden_weights=True)


if __name__ == "__main__":
    ex_3_1()

    # ex_3_2([])

    # ex_3_2_greedy()
