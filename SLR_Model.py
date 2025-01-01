import keras
from keras import Layer
from keras.src.layers import Input, Dense, LSTM, Dropout, BatchNormalization, Masking, Bidirectional, \
    GlobalAveragePooling1D, GRU, Attention, ReLU, Add, MultiHeadAttention, MaxPooling1D, Conv1D, Activation, \
    LayerNormalization, SimpleRNN
from keras.src.models import Model
from keras.src.regularizers import L2, L1L2


class SLR_Model:

    def __init__(self, input_shape, word_labels, mask_value):
        self.input_shape = input_shape
        self.word_labels = word_labels
        self.mask_value = mask_value

        self.model = self.build_model()

    def build_model(self, activation="relu"):
        data = Input(name='input', shape=self.input_shape)

        # LSTM Layer
        x = LSTM(units=128, return_sequences=True, activation='tanh')(data)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # GRU Layer
        x = GRU(units=256, return_sequences=False, activation='tanh')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Dense Layer for feature extraction
        x = Dense(units=256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        output = Dense(self.word_labels.shape[0], activation="softmax")(x)

        model = Model(inputs=data, outputs=output)

        return model

    def compile_model(self):
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss = 'categorical_crossentropy'
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