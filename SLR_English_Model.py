import keras
from keras import Layer
from keras.src import ops
from keras.src.layers import Input, Dense, LSTM, Dropout, BatchNormalization, Masking, Bidirectional, \
    GlobalAveragePooling1D, GRU, Attention, ReLU, Add, MultiHeadAttention, MaxPooling1D, Conv1D, Activation, \
    LayerNormalization, SimpleRNN, GaussianNoise
from keras.src.models import Model
from keras.src.regularizers import L2, L1L2


class SLR_English_Model:

    def __init__(self, input_shape, word_labels, mask_value):
        self.input_shape = input_shape
        self.word_labels = word_labels
        self.mask_value = mask_value

        self.model = self.build_model()

    def tcn_block(self, x, filters, k=5, d=1, p=0.1):
        y = Conv1D(filters, k, padding='same', dilation_rate=d, kernel_regularizer=L2(1e-4))(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Dropout(p)(y)
        y = Conv1D(filters, k, padding='same', dilation_rate=d, kernel_regularizer=L2(1e-4))(y)
        y = BatchNormalization()(y)
        x = Add()([x, y])
        return Activation('relu')(x)

    def attn_pool(self, x, mask, units=128):
        scores = Dense(units, activation='tanh')(x)
        scores = Dense(1)(scores)
        if mask is not None:
            large_neg = ops.cast(-1e9, scores.dtype)
            scores = ops.where(mask, scores, large_neg)
        w = ops.softmax(scores, axis=1)
        return ops.sum(w * x, axis=1)

    def build_model(self, activation="relu"):
        inp = Input(name='input', shape=self.input_shape)  # (100,190)
        x = Masking(mask_value=self.mask_value)(inp)
        x = GaussianNoise(0.01)(x)

        mask = ops.any(ops.not_equal(inp, self.mask_value), axis=-1, keepdims=True)

        x = Dense(256, activation=activation)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        for d in [1, 2, 4, 8]:
            x = self.tcn_block(x, 256, k=5, d=d, p=0.1)

        x = Bidirectional(GRU(192, return_sequences=True, activation='tanh'))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.4)(x)

        pooled = self.attn_pool(x, mask)

        x = Dense(256, activation=activation)(pooled)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        out = Dense(self.word_labels.shape[0], activation='softmax')(x)

        return Model(inp, out)

    def compile_model(self):
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        metrics = ['categorical_accuracy']
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def train(self, keypoints, num_labels, epochs, callbacks, x_val, y_val, batch_size, class_weight):
        self.model.fit(keypoints, num_labels, epochs=epochs, callbacks=callbacks, validation_data=(x_val, y_val), batch_size=batch_size, class_weight=class_weight)

    def validate(self, val_keypoints, val_num_labels):
        return self.model.evaluate(val_keypoints, val_num_labels)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def summary(self, line_length):
        return self.model.summary(line_length=line_length)