from keras import metrics
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import Utils
import numpy as np
from autoenconder import AutoEncoder, ErrorsCallback
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout
from sklearn import metrics
import time
from keras import backend as K


def ex_3_2_expeirment():
    x_train = np.loadtxt('binMNIST_data/bindigit_trn.csv', delimiter=',', dtype=float)
    x_test = np.loadtxt('binMNIST_data/bindigit_tst.csv', delimiter=',', dtype=float)

    y_train = np.loadtxt('binMNIST_data/targetdigit_trn.csv', delimiter=',', dtype=float)
    y_test = np.loadtxt('binMNIST_data/targetdigit_tst.csv', delimiter=',', dtype=float)

    target_train = to_categorical(y_train, num_classes=10)
    target_test = to_categorical(y_test, num_classes=10)


    input_img = Input(shape=(784,))
    distorted_input1 = Dropout(.1)(input_img)
    encoded1 = Dense(800, activation='sigmoid')(distorted_input1)
    encoded1_bn = BatchNormalization()(encoded1)
    decoded1 = Dense(784, activation='sigmoid')(encoded1_bn)

    autoencoder1 = Model(inputs=input_img, outputs=decoded1)
    encoder1 = Model(inputs=input_img, outputs=encoded1_bn)

    # Layer 2
    encoded1_input = Input(shape=(800,))
    distorted_input2 = Dropout(.2)(encoded1_input)
    encoded2 = Dense(400, activation='sigmoid')(distorted_input2)
    encoded2_bn = BatchNormalization()(encoded2)
    decoded2 = Dense(800, activation='sigmoid')(encoded2_bn)

    autoencoder2 = Model(inputs=encoded1_input, outputs=decoded2)
    encoder2 = Model(inputs=encoded1_input, outputs=encoded2_bn)

    # Layer 3 - which we won't end up fitting in the interest of time
    encoded2_input = Input(shape=(400,))
    distorted_input3 = Dropout(.3)(encoded2_input)
    encoded3 = Dense(200, activation='sigmoid')(distorted_input3)
    encoded3_bn = BatchNormalization()(encoded3)
    decoded3 = Dense(400, activation='sigmoid')(encoded3_bn)

    autoencoder3 = Model(inputs=encoded2_input, outputs=decoded3)
    encoder3 = Model(inputs=encoded2_input, outputs=encoded3_bn)

    # Deep Autoencoder
    encoded1_da = Dense(800, activation='sigmoid')(input_img)
    encoded1_da_bn = BatchNormalization()(encoded1_da)
    encoded2_da = Dense(400, activation='sigmoid')(encoded1_da_bn)
    encoded2_da_bn = BatchNormalization()(encoded2_da)
    encoded3_da = Dense(200, activation='sigmoid')(encoded2_da_bn)
    encoded3_da_bn = BatchNormalization()(encoded3_da)
    decoded3_da = Dense(400, activation='sigmoid')(encoded3_da_bn)
    decoded2_da = Dense(800, activation='sigmoid')(decoded3_da)
    decoded1_da = Dense(784, activation='sigmoid')(decoded2_da)

    deep_autoencoder = Model(inputs=input_img, outputs=decoded1_da)

    # Not as Deep Autoencoder
    nad_encoded1_da = Dense(800, activation='sigmoid')(input_img)
    nad_encoded1_da_bn = BatchNormalization()(nad_encoded1_da)
    nad_encoded2_da = Dense(400, activation='sigmoid')(nad_encoded1_da_bn)
    nad_encoded2_da_bn = BatchNormalization()(nad_encoded2_da)
    nad_decoded2_da = Dense(800, activation='sigmoid')(nad_encoded2_da_bn)
    nad_decoded1_da = Dense(784, activation='sigmoid')(nad_decoded2_da)

    nad_deep_autoencoder = Model(inputs=input_img, outputs=nad_decoded1_da)

    sgd1 = SGD(lr=5, decay=0.5, momentum=.85, nesterov=True)
    sgd2 = SGD(lr=5, decay=0.5, momentum=.85, nesterov=True)
    sgd3 = SGD(lr=5, decay=0.5, momentum=.85, nesterov=True)

    autoencoder1.compile(loss='binary_crossentropy', optimizer=sgd1)
    autoencoder2.compile(loss='binary_crossentropy', optimizer=sgd2)
    autoencoder3.compile(loss='binary_crossentropy', optimizer=sgd3)

    encoder1.compile(loss='binary_crossentropy', optimizer=sgd1)
    encoder2.compile(loss='binary_crossentropy', optimizer=sgd1)
    encoder3.compile(loss='binary_crossentropy', optimizer=sgd1)

    deep_autoencoder.compile(loss='binary_crossentropy', optimizer=sgd1)
    nad_deep_autoencoder.compile(loss='binary_crossentropy', optimizer=sgd1)

    autoencoder1.fit(x_train, x_train,
                     epochs=8, batch_size=512,
                     validation_split=0.3,
                     shuffle=True)

    first_layer_code = encoder1.predict(x_train)
    print(first_layer_code.shape)

    autoencoder2.fit(first_layer_code, first_layer_code,
                     epochs=8, batch_size=512,
                     validation_split=0.25,
                     shuffle=True)

    second_layer_code = encoder2.predict(first_layer_code)
    print(second_layer_code.shape)

    autoencoder3.fit(second_layer_code, second_layer_code,
                     epochs=8, batch_size=512,
                     validation_split=0.30,
                     shuffle=True)

    # Setting the weights of the deep autoencoder
    # deep_autoencoder.layers[1].set_weights(autoencoder1.layers[2].get_weights()) # first dense layer
    # deep_autoencoder.layers[2].set_weights(autoencoder1.layers[3].get_weights()) # first bn layer
    # deep_autoencoder.layers[3].set_weights(autoencoder2.layers[2].get_weights()) # second dense layer
    # deep_autoencoder.layers[4].set_weights(autoencoder2.layers[3].get_weights()) # second bn layer
    # deep_autoencoder.layers[5].set_weights(autoencoder3.layers[2].get_weights()) # thrird dense layer
    # deep_autoencoder.layers[6].set_weights(autoencoder3.layers[3].get_weights()) # third bn layer
    # deep_autoencoder.layers[7].set_weights(autoencoder3.layers[4].get_weights()) # first decoder
    # deep_autoencoder.layers[8].set_weights(autoencoder2.layers[4].get_weights()) # second decoder
    # deep_autoencoder.layers[9].set_weights(autoencoder1.layers[4].get_weights()) # third decoder

    # Setting up the weights of the not-as-deep autoencoder
    nad_deep_autoencoder.layers[1].set_weights(autoencoder1.layers[2].get_weights())  # first dense layer
    nad_deep_autoencoder.layers[2].set_weights(autoencoder1.layers[3].get_weights())  # first bn layer
    nad_deep_autoencoder.layers[3].set_weights(autoencoder2.layers[2].get_weights())  # second dense layer
    nad_deep_autoencoder.layers[4].set_weights(autoencoder2.layers[3].get_weights())  # second bn layer
    nad_deep_autoencoder.layers[5].set_weights(autoencoder2.layers[4].get_weights())  # second decoder
    nad_deep_autoencoder.layers[6].set_weights(autoencoder1.layers[4].get_weights())  # third decoder



    dense1 = Dense(500, activation='relu')(nad_decoded1_da)
    # dense1 = Dense(500, activation='relu')(decoded1_da)
    # dense1_drop = Dropout(.3)(dense1)
    # dense1_bn = BatchNormalization()(dense1_drop)
    dense2 = Dense(10, activation='sigmoid')(dense1)

    classifier = Model(inputs=input_img, outputs=dense2)
    sgd4 = SGD(lr=.1, decay=0.001, momentum=.95, nesterov=True)
    classifier.compile(loss='categorical_crossentropy', optimizer=sgd4, metrics=['accuracy'])

    classifier.fit(x_train, target_train,
                   epochs=15, batch_size=600,
                   validation_split=0.25,
                   shuffle=True)

    test_preds = classifier.predict(x_test)
    predictions = np.argmax(test_preds, axis=1)
    true_digits = np.argmax(target_test, axis=1)

    n_correct = np.sum(np.equal(predictions, true_digits).astype(int))
    total = float(len(predictions))
    print("Test Accuracy:", round(n_correct / total, 3))


def ex_3_2(num_of_nodes_in_hidden_layers):
    x_train = np.loadtxt('binMNIST_data/bindigit_trn.csv', delimiter=',', dtype=float)
    x_test = np.loadtxt('binMNIST_data/bindigit_tst.csv', delimiter=',', dtype=float)

    y_train = np.loadtxt('binMNIST_data/targetdigit_trn.csv', delimiter=',', dtype=float)
    y_test = np.loadtxt('binMNIST_data/targetdigit_tst.csv', delimiter=',', dtype=float)

    target_train = to_categorical(y_train, num_classes=10)

    n_epochs = 8
    batch_size = 512

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

    final_model.add(Dense(10, activation='sigmoid'))

    final_model.compile(optimizer=SGD(lr=.1, decay=0.001, momentum=.95, nesterov=True), loss='mean_squared_error',
                        metrics=['accuracy'])

    history = final_model.fit(x_train, target_train,
                              epochs=n_epochs,
                              batch_size=batch_size,
                              shuffle=True, validation_split=0.25)

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
    # ex_3_2([512,256, 128])

    ex_3_2_expeirment()