import keras
from keras.src.layers import Input, Dense, LSTM, Dropout, BatchNormalization, Masking
from keras.src.models import Model


class SLR_Model:

    def __init__(self, input_shape, word_labels, mask_value):
        self.input_shape = input_shape
        self.word_labels = word_labels
        self.mask_value = mask_value

        self.model = self.build_model()

    def LSTM_Block(self, x, units_num, return_seq, activ):
        x = LSTM(units=units_num, return_sequences=return_seq, activation=activ)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        return x

    def build_model(self, activation="leaky_relu"):
        data = Input(name='input', shape=self.input_shape)

        masked_data = Masking(mask_value=self.mask_value)(data)

        lstm_block1 = self.LSTM_Block(x=masked_data, units_num=64, return_seq=True, activ=activation)
        lstm_block2 = self.LSTM_Block(x=lstm_block1, units_num=128, return_seq=True, activ=activation)
        lstm_block3 = self.LSTM_Block(x=lstm_block2, units_num=64, return_seq=True, activ=activation)
        lstm_block4 = self.LSTM_Block(x=lstm_block3, units_num=32, return_seq=False, activ=activation)

        dense = Dense(64, activation=activation)(lstm_block4)
        output = Dense(self.word_labels.shape[0], activation='softmax')(dense)

        model = Model(inputs=data, outputs=output)

        return model

    def compile_model(self):
        optimizer = 'Adam'
        loss = 'categorical_crossentropy'
        metrics = ['categorical_accuracy']
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def train(self, train, validate, epochs, callbacks):
        self.model.fit(train, validate, epochs=epochs, callbacks=callbacks)

    def validate(self, val_keypoints, val_num_labels):
        return self.model.evaluate(val_keypoints, val_num_labels)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def summary(self, line_length):
        return self.model.summary(line_length=line_length)