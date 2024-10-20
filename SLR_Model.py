import tensorflow as tf

try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]

except: pass

class SLR_Model:

    def __init__(self):
        self.model = self.build_model()

    def build_model(self, activation="leaky_relu", dropout=0.1):
        xxxxxxx

        model = Model(inputs=data, outputs=output)

        return model

    def compile_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.model.compile(loss=xxxxx, optimizer=optimizer, metrics=xxxxx, run_eagerly=xxxxx)

    def train(self, train, validate, epochs, workers, callbacks):
        self.model.fit(train, validation_data=validate, epochs=epochs, workers=workers, callbacks=callbacks)

    def validate(self, validate):
        return self.model.evaluate(validate)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def summary(self, line_length):
        return self.model.summary(line_length=line_length)