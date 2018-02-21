from keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, LeakyReLU, Dropout
from keras import optimizers, losses, activations
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras import backend as K


class CNNModel(object):
    def __init__(self, input_shape, cnn_filters, cnn_kernels, cnn_pools, dense_units, dropout=0.5, leaky_alpha=0.3):
        self.input_shape = input_shape
        self.num_layers = len(cnn_filters)
        self.num_dense = len(dense_units)
        self.filters = cnn_filters
        self.kernels = cnn_kernels
        self.poolings = cnn_pools
        self.dense_units = dense_units
        self.dropout = dropout
        self.leaky_alpha = leaky_alpha
        self.model = None


        # GPU Config
        # cfg = K.tf.ConfigProto()
        # cfg.gpu_options.allow_growth = True
        # K.set_session(K.tf.Session(config=cfg))

    def build_model(self, optimizer='adam', loss='binary_crossentropy'):
        print("Creating CNN Model....")
        input_ = Input(shape=self.input_shape)
        print(input_)

        x = Conv1D(filters=self.filters[0],
                   kernel_size=self.kernels[0])(input_)
        x = LeakyReLU(alpha=self.leaky_alpha)(x)
        x = Dropout(rate=self.dropout)(x)

        for i in range(1, self.num_layers):
            x = Conv1D(filters=self.filters[i], kernel_size=self.kernels[i])(x)
            x = LeakyReLU(alpha=self.leaky_alpha)(x)
            x = Dropout(rate=self.dropout)(x)
            if self.poolings[i]:
                x = MaxPooling1D(pool_size=self.poolings[i])(x)

        x = Flatten()(x)

        for i in range(self.num_dense):
            x = Dense(units=self.dense_units[i])(x)
            x = Dropout(rate=self.dropout)(x)

        x = Dense(units=1, activation='sigmoid')(x)

        self.model = Model(inputs=input_, outputs=x)
        self.model.compile(optimizer=optimizer, loss=loss,
                           metrics=['accuracy'])
        self.model.summary()

    def train(self, inputs, labels, epochs=20, batch_size=50, validation_split=0.3):
        print("Training Model....")
        self.model.fit(x=inputs, y=labels, epochs=epochs,
                       batch_size=batch_size, validation_split=validation_split)

        print("Saving Model.....")
        self.model.save('char_cnn.h5')
