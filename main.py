import os
import re

from GUI_Language_Selector import Language_Selector
from Graphs import Graphs

os.environ["KERAS_BACKEND"] = "tensorflow"
from Data_Processing import DataProcessing
from GUI import GUI
from TrainModel import TrainModel


if __name__ == "__main__":

    create_dataset = False
    convert_data = False
    add_padding = False
    save_data = False
    train_model = False
    run_app = True

    language_selector = Language_Selector()
    chosen_language = language_selector.language

    data_processor = DataProcessing(chosen_language)

    base_lang_path = os.path.join("Data", chosen_language)
    dataset_videos_classes = os.path.join(base_lang_path, "Dataset", "Videos")
    dataset_keypoints_classes = os.path.join(base_lang_path, "Dataset", "Keypoints")

    if create_dataset:
        print("Creating Dataset...")
        data_processor.create_folders()
        data_processor.move_videos()
        data_processor.count_and_move_videos(dataset_videos_classes)
        print("Finished Creating Dataset...")

    if convert_data:
        print("Converting Data...")
        data_processor.convert_videos_to_numpy(dataset_videos_classes, dataset_keypoints_classes)
        print("Finished Converting Data...")

    if add_padding:
        print("Adding Padding to Data...")
        max_frames = data_processor.get_max_files_in_subfolder()
        data_processor.add_padding(max_frames, dataset_keypoints_classes)
        print("Finished Adding Padding to Data...")

    if save_data:
        print("Saving Data...")
        max_frames = data_processor.get_max_files_in_subfolder()
        word_labels, num_labels, keypoints = data_processor.get_dataset_vocab(dataset_keypoints_classes)
        data_processor.save_dataset_to_files(word_labels, num_labels, keypoints, max_frames,"dataset")
        print("Finished Saving Data...")

    if train_model:
        print("Training Model...")
        word_labels, max_frames, keypoints, num_labels = data_processor.load_dataset_from_files()
        TrainModel(word_labels, max_frames, keypoints, num_labels, chosen_language)

        print("Finished Training Model...")

    if run_app:
        print("Running App...")
        logs_path = os.path.join("Model", chosen_language, "logs.log")
        Graphs(logs_path)

        app = GUI(data_processor, chosen_language)
        app.run()
