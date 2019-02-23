from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.initializers import RandomNormal


class AutoEncoder(object):
    def __init__(self, input_dim, encode_dim, l2_value=0, encode_activation='relu'):
        self.input_dim = input_dim
        self.encode_dim = encode_dim  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
        self.regularizer = regularizers.l2(l2_value)
        self.encode_activation = encode_activation
        self._create_model()

    def train(self, x_train, x_test, n_epochs=50, batch_size=256, loss='binary_crossentropy', optimizer='adadelta'):
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.fit(x_train, x_train,
                       epochs=n_epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_data=(x_test, x_test))

    def _create_model(self):

        original_input = Input(shape=(self.input_dim,))
        encoded = self._create_encoded_layer(original_input)
        decoded = self._create_decoded_layer(encoded)

        autoencoder = Model(original_input, decoded)

        encoder = Model(original_input, encoded)

        encoder_input = Input(shape=self.encode_dim)
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(decoder_layer, decoder_layer(encoder_input))

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
