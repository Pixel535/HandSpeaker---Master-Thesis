import os
import numpy as np
os.environ["KERAS_BACKEND"] = "tensorflow"
from Data_Processing import DataProcessing
from GUI import GUI
from TrainModel import TrainModel


if __name__ == "__main__":

    convert_data = False
    save_data = False
    is_model_trained = False
    run_app = False

    data_processor = DataProcessing()

    if convert_data:
        print("Converting Data...")
        data_processor.convert_videos_to_numpy()
        print("Finished Converting Data...")

    if save_data:
        print("Saving Data...")
        word_labels, num_labels, keypoints = data_processor.get_dataset_vocab()
        data_processor.save_dataset_to_files(word_labels, num_labels, keypoints)
        print("Finished Saving Data...")

    word_labels, num_labels, keypoints = data_processor.load_dataset_from_files()
    print(word_labels)
    print(num_labels)

    if not is_model_trained:
        print("Training Model...")
        train_model = TrainModel(word_labels, num_labels, keypoints)
        print("Finished Training Model...")

    if run_app:
        print("Running App...")
        #logs_path = "Model/logs.log"
        #Graphs(logs_path)

        app = GUI(data_processor)
        app.run()

