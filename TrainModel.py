import os
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from Callbacks_mltu import TrainLogger
from SLR_Model import SLR_Model
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, BackupAndRestore

class TrainModel:
    def __init__(self, word_labels, max_frames, keypoints, num_labels, lang):

        self.lang = lang
        self.word_labels = word_labels

        self.keypoints = keypoints
        self.num_labels = num_labels

        self.train_keypoints = None
        self.train_num_labels = None
        self.val_keypoints = None
        self.val_num_labels = None

        self.epochs = 5000
        self.mask_value = 0.0
        self.batch_size = 32
        self.max_frames = max_frames
        self.SLR_Model = SLR_Model

        self.trainModel()

    def trainModel(self):

        print("Loading Data...")

        self.train_keypoints, self.val_keypoints, self.train_num_labels, self.val_num_labels = train_test_split(self.keypoints, self.num_labels, test_size=0.1, random_state=34, stratify=self.num_labels)

        integer_train_labels = np.argmax(self.train_num_labels, axis=1)
        class_counts = Counter(integer_train_labels)
        total_samples = len(integer_train_labels)
        num_classes = len(self.word_labels)
        class_weights = {cls: total_samples / (count * num_classes) for cls, count in class_counts.items()}

        print("Finished Loading Data...")

        print("Creating Model...")

        model_dir = os.path.join("Model", self.lang)
        model_path = os.path.join(model_dir, "model.keras")
        logs_path = os.path.join(model_dir, "logs")

        if os.path.exists(model_path):
            input_shape = (self.max_frames, 126)
            SLR_Model = self.SLR_Model(input_shape, self.word_labels, self.mask_value)
            SLR_Model.compile_model()
            SLR_Model.model.load_weights("Model/model.keras")
            print(SLR_Model.model.get_weights())
            new_model = False
        else:
            input_shape = (self.max_frames, 126)
            SLR_Model = self.SLR_Model(input_shape, self.word_labels, self.mask_value)
            SLR_Model.compile_model()
            SLR_Model.summary(line_length=110)
            print(SLR_Model.model.get_weights())
            new_model = True

        print("Model Created...")

        print("Training Model...")

        earlystopper = EarlyStopping(monitor='val_categorical_accuracy', patience=30, verbose=1, mode='max')
        checkpoint = ModelCheckpoint(model_path, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
        trainLogger = TrainLogger("Model")
        tb_callback = TensorBoard(logs_path, update_freq=1)
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.9, min_delta=1e-10, patience=10, verbose=1,
                                           mode='auto')

        is_Trained = False

        if is_Trained is False:
            if new_model is True:
                SLR_Model.train(self.train_keypoints,
                              self.train_num_labels,
                              epochs=self.epochs,
                              callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback],
                              x_val=self.val_keypoints,
                              y_val=self.val_num_labels,
                              batch_size=self.batch_size,
                              class_weight=class_weights)
            else:
                SLR_Model.train(self.train_keypoints,
                                self.train_num_labels,
                                epochs=self.epochs,
                                callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback],
                                x_val=self.val_keypoints,
                                y_val=self.val_num_labels,
                                batch_size=self.batch_size,
                                class_weight=class_weights)


        print("Finished Training Model...")
        loss, categorical_accuracy = SLR_Model.validate(self.val_keypoints, self.val_num_labels)

        print("Validation loss: ", loss)
        print("Validation Categorical Accuracy: ", categorical_accuracy)
