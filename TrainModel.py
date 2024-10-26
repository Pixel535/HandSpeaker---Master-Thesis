import os
import keras
import numpy as np
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from Callbacks_mltu import Model2onnx, TrainLogger
from SLR_Model import SLR_Model
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, BackupAndRestore

class TrainModel:
    def __init__(self, word_labels, num_labels, keypoints):

        self.word_labels = word_labels
        self.num_labels = num_labels
        self.keypoints = keypoints
        self.train_ratio = 0.2
        self.epochs = 1000
        self.mask_value = -999.0
        self.SLR_Model = SLR_Model

        self.trainModel()

    def trainModel(self):

        print("Loading Data...")

        keypoints = self.keypoints

        local_mins = [np.min(sequence) for video in keypoints for sequence in video]
        global_min = min(local_mins)

        if global_min < self.mask_value:
            self.mask_value = global_min - self.mask_value

        max_frames = max(len(kp) for kp in keypoints)
        for seq in keypoints:
            while len(seq) < max_frames:
                seq.append(np.full((126,), self.mask_value))

        keypoints, num_labels = np.array(keypoints), to_categorical(self.num_labels).astype(int)

        print("Finished Loading Data...")

        print("Creating Model...")

        if os.path.exists("Model/model.keras"):
            input_shape = (max_frames, 126)
            SLR_Model = self.SLR_Model(input_shape, self.word_labels, self.mask_value)
            SLR_Model.compile_model()
            SLR_Model.model.load_weights("Model/model.keras")
            print(SLR_Model.model.get_weights())
            new_model = False
        else:
            input_shape = (max_frames, 126)
            SLR_Model = self.SLR_Model(input_shape, self.word_labels, self.mask_value)
            SLR_Model.compile_model()
            SLR_Model.summary(line_length=110)
            print(SLR_Model.model.get_weights())
            new_model = True

        print("Model Created...")

        print("Training Model...")

        training_keypoints, val_keypoints, training_num_labels, val_num_labels = train_test_split(keypoints, num_labels,
                                                                                                  test_size=self.train_ratio,
                                                                                                  random_state=34,
                                                                                                  stratify=num_labels)

        backup = BackupAndRestore(backup_dir="Model/backup")
        earlystopper = EarlyStopping(monitor='val_CER', patience=20, verbose=1, mode='min')
        checkpoint = ModelCheckpoint("Model/model.keras", monitor='val_CER', verbose=1, save_best_only=True, mode='min')
        trainLogger = TrainLogger("Model")
        tb_callback = TensorBoard('Model/logs', update_freq=1)
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_CER', factor=0.9, min_delta=1e-10, patience=10, verbose=1,
                                           mode='auto')
        model2onnx = Model2onnx("Model/model.keras")

        is_Trained = False

        if is_Trained is False:
            if new_model is True:
                SLR_Model.train(training_keypoints,
                                training_num_labels,
                                epochs=self.epochs,
                                callbacks=[backup, earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx])
            else:
                SLR_Model.train(training_keypoints,
                                training_num_labels,
                                epochs=self.epochs,
                                callbacks=[backup, earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx])

        print("Finished Training Model...")

        loss, categorical_crossentropy = SLR_Model.validate(val_keypoints, val_num_labels)

        print("Validation loss: ", loss)
        print("Validation Categorical Crossentropy: ", categorical_crossentropy)

        training_keypoints.to_csv(os.path.join("Test Data", "training_keypoints.csv"))
        training_num_labels.to_csv(os.path.join("Test Data", "training_num_labels.csv"))
        val_keypoints.to_csv(os.path.join("Test Data", "val_keypoints.csv"))
        val_num_labels.to_csv(os.path.join("Test Data", "val_num_labels.csv"))