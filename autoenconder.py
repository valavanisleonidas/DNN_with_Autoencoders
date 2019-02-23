from keras.callbacks import Callback
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.initializers import RandomNormal
from keras.optimizers import SGD


class AutoEncoder(object):
    def __init__(self, input_dim, encode_dim, callbacks, l2_value=0, encode_activation='relu'):
        self.input_dim = input_dim
        self.encode_dim = encode_dim  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
        self.regularizer = regularizers.l2(l2_value)
        self.encode_activation = encode_activation
        self._create_model()
        self.callbacks = callbacks.append(ErrorsCallback)

    def train(self, x_train, x_test, n_epochs=50, batch_size=256, loss='binary_crossentropy',
              optimizer=SGD(lr=0.1, momentum=0.9)):
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.fit(x_train, x_train,
                       epochs=n_epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_data=(x_test, x_test),
                       callbacks=self.callbacks
                       )

    def predict(self, x_test):
        encoded_imgs = self.encoder.predict(x_test)
        decoded_imgs = self.decoder.predict(encoded_imgs)
        return decoded_imgs

    def _create_model(self):
        original_input = Input(shape=(self.input_dim,))
        encoded = self._create_encoded_layer(original_input)
        decoded = self._create_decoded_layer(encoded)

        autoencoder = Model(original_input, decoded)

        encoder = Model(original_input, encoded)

        encoder_input = Input(shape=(self.encode_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoder_input, decoder_layer(encoder_input))

        self.encoder = encoder
        self.decoder = decoder
        self.model = autoencoder

    def _create_encoded_layer(self, original_input):
        encoded = Dense(self.encode_dim,
                        activation=self.encode_activation,
                        kernel_initializer=RandomNormal(mean=0, stddev=0.01, seed=123),
                        kernel_regularizer=self.regularizer,
                        bias_initializer='zeros')(original_input)
        return encoded

    def _create_decoded_layer(self, encoded):
        decoded = Dense(self.input_dim, activation='sigmoid', bias_initializer='zeros')(encoded)
        return decoded


class ErrorsCallback(Callback):
    def __init__(self, val_in, val_out, train_in, train_out, test_in, test_out):
        self.val_in = val_in
        self.val_out = val_out
        self.train_in = train_in
        self.train_out = train_out
        self.test_in = test_in
        self.test_out = test_out
        self.mse_train = []
        self.mse_val = []
        self.mse_test = []

    def on_epoch_end(self, epoch, logs={}):
        self.mse_val.append(self.model.evaluate(self.val_in, self.val_out, verbose=0))
        self.mse_train.append(self.model.evaluate(self.train_in, self.train_out, verbose=0))
        self.mse_test.append(self.model.evaluate(self.test_in, self.test_out, verbose=0))
