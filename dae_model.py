from keras.layers import Input, Dense
from keras.models import Model


class DAE(object):
    def __init__(self, num_words, encoder_layers, decoder_layers):

        self.num_words = num_words
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.input_shape = Input(shape=(self.num_words,))

        # encoder's layers
        self.encoded = Dense(self.encoder_layers[0], activation="relu")(
            self.input_shape
        )
        for encoder_layer in self.encoder_layers[1:]:
            self.encoded = Dense(encoder_layer, activation="relu")(self.encoded)

        # decoder's layers
        self.decoded = Dense(self.decoder_layers[0], activation="relu")(self.encoded)
        for decoder_layer in self.decoder_layers[1:]:
            self.decoded = Dense(decoder_layer, activation="relu")(self.decoded)

        self.decoded = Dense(num_words, activation="sigmoid")(self.decoded)

        self.model = Model(self.input_shape, self.decoded)
