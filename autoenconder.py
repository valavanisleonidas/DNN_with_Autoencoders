from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers



class AutoEncoder(object):
    def __init__(self, input_dim, encode_dim, l2_value=0, encode_activation='relu'):
        self.input_dim = input_dim
        self.encode_dim = encode_dim  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
        self.regularizer = regularizers.l2(l2_value)
        self.encode_activation = encode_activation
        self._create_model()

    def _create_model(self):

        input = Input(shape=(self.input_dim,))
        # "encoded" is the encoded representation of the input
        encoder = Dense(self.encode_dim, activation=self.encode_activation, kernel_regularizer=self.regularizer)(
            input)
        # "decoded" is the lossy reconstruction of the input
        decoder = Dense(self.input_dim, activation='sigmoid')(encoder)
        # this model maps an input to its reconstruction
        model = Model(input, decoder)

        self.encoder = encoder
        self.decoder = decoder
        self.model = model

    def train(self, x_train, x_test, n_epochs=50, batch_size=256, loss='binary_crossentropy', optimizer='adadelta'):
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.fit(x_train, x_train,
                       epochs=n_epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_data=(x_test, x_test))



