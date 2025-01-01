import random
import shutil
from collections import defaultdict
import pandas as pd
import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt


class DataProcessing:

    def __init__(self):
        self.dataset_videos_classes = "Data/Dataset/Videos"
        self.dataset_keypoints_classes = "Data/Dataset/Keypoints"

        self.dataset_videos_rejected = "Data/Dataset/Rejected Videos"
        self.dataset_downloaded_videos = "Data/Dataset/Downloaded Videos"

        self.dictionary_path = "Data/Dictionary"
        self.files_path = "Data/Files"

        self.dataset_test_file_path = "Data/Dataset/test.csv"
        self.dataset_train_file_path = "Data/Dataset/train.csv"
        self.dataset_val_file_path = "Data/Dataset/val.csv"

        self.holistic_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mask_value = 0.0

    def list_video_folders_in_directory(self):
        folders = [name for name in os.listdir(self.dataset_videos_classes) if os.path.isdir(os.path.join(self.dataset_videos_classes, name))]
        return folders

    def get_vocab_and_dict_for_Text_to_SL(self):
        vocab = []
        file_dict = {}
        for file in os.listdir(self.dictionary_path):
            file_path = os.path.join(self.dictionary_path, file)
            file_name, file_extension = os.path.splitext(file)
            vocab.append(file_name)
            file_dict[file_name] = file_path
        return vocab, file_dict

    def count_and_move_videos(self, root_folder):
        count = 0

        os.makedirs(self.dataset_videos_rejected, exist_ok=True)

        for subdir, _, files in os.walk(root_folder):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    file_path = os.path.join(subdir, file)
                    try:
                        video = cv2.VideoCapture(file_path)
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()

                        if frame_count > 200:
                            print(f"Moving: {file_path}")
                            shutil.move(file_path, os.path.join(self.dataset_videos_rejected, file))
                            count += 1
                    except Exception as e:
                        print(f"[Error] {file_path}: {e}")

        return count

    def create_folders(self):
        self.create_folders_from_classes(self.dataset_videos_classes, self.dataset_test_file_path)
        self.create_folders_from_classes(self.dataset_videos_classes, self.dataset_train_file_path)
        self.create_folders_from_classes(self.dataset_videos_classes, self.dataset_val_file_path)

        self.create_folders_from_classes(self.dataset_keypoints_classes, self.dataset_test_file_path)
        self.create_folders_from_classes(self.dataset_keypoints_classes, self.dataset_train_file_path)
        self.create_folders_from_classes(self.dataset_keypoints_classes, self.dataset_val_file_path)

    def move_videos(self):
        self.move_videos_to_folder(self.dataset_videos_classes, self.dataset_test_file_path)
        self.move_videos_to_folder(self.dataset_videos_classes, self.dataset_train_file_path)
        self.move_videos_to_folder(self.dataset_videos_classes, self.dataset_val_file_path)

    def move_videos_to_folder(self, output_dir, file_path):
        df = pd.read_csv(file_path)

        for index, row in df.iterrows():
            video_file = row['Video file']
            gloss_folder = row['Gloss']

            source_path = os.path.join(self.dataset_downloaded_videos, video_file)

            target_folder = os.path.join(output_dir, gloss_folder)
            target_path = os.path.join(target_folder, video_file)

            print(f"Processing row {index + 1}: {video_file}")

            if os.path.exists(source_path):
                shutil.move(source_path, target_path)
                print(f"Moved: {video_file} to {target_folder}")
            else:
                print(f"File not found: {source_path}")

    def create_folders_from_classes(self, output_dir, file_path):
        df = pd.read_csv(file_path)
        df['Gloss'] = df.apply(self.fix_gloss, axis=1)

        df.to_csv(file_path, index=False)

        unique_glosses = df['Gloss'].unique()

        for gloss in unique_glosses:
            folder_path = os.path.join(output_dir, gloss)
            try:
                os.makedirs(folder_path, exist_ok=True)
                print(f"Folder created successfully: {folder_path}")
            except OSError as e:
                print(f"Error creating folder {folder_path}: {e}")


    def fix_gloss(self, row):
        gloss = row['Gloss']
        video_gloss = row['Video file'].split('-')[1].replace(".mp4", "").strip()
        if video_gloss.lower().startswith("seed"):
            video_gloss = video_gloss[4:]
        video_gloss = video_gloss.replace("_", " ")
        if gloss.upper() == video_gloss.replace(" ", "").upper():
            result = video_gloss
        else:
            result = gloss
        return result


    def image_processing(self, image, model):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return results, image

    def draw_landmarks(self, image, results):
        mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

    def keypoint_extraction(self, results):
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
        keypoints = np.concatenate([lh, rh])
        return keypoints

    def get_max_files_in_subfolder(self):
        max_files_count = 0

        for root_folder in os.listdir(self.dataset_keypoints_classes):
            root_path = os.path.join(self.dataset_keypoints_classes, root_folder)

            if os.path.isdir(root_path):
                for subfolder in os.listdir(root_path):
                    subfolder_path = os.path.join(root_path, subfolder)

                    if os.path.isdir(subfolder_path):
                        files_count = len([file for file in os.listdir(subfolder_path) if
                                           os.path.isfile(os.path.join(subfolder_path, file))])

                        if files_count > max_files_count:
                            max_files_count = files_count

        print("Finished Scanning Keypoints Directory")

        return max_files_count

    def add_padding(self, max_files_count, input_path):

        for word_folder in os.listdir(input_path):
            word_folder_path = os.path.join(input_path, word_folder)

            if os.path.isdir(word_folder_path):
                word = word_folder
                for videoFolder in os.listdir(word_folder_path):
                    video_folder_path = os.path.join(word_folder_path, videoFolder)

                    if os.path.isdir(video_folder_path):
                        files = [file for file in os.listdir(video_folder_path) if file.endswith('.npy')]
                        current_file_count = len(files)

                        files_to_add = max_files_count - current_file_count
                        sequence_folder_name = os.path.basename(video_folder_path)
                        folder_count = int(sequence_folder_name.split('_')[1])
                        start_frame_num = current_file_count

                        for i in range(files_to_add):
                            dummy_data = np.full((126,), self.mask_value)
                            frame_num = start_frame_num + i
                            new_file_name = f'w_{word}_s_{folder_count}_f_{frame_num}.npy'
                            new_file_path = os.path.join(video_folder_path, new_file_name)
                            np.save(new_file_path, dummy_data)

            print(f"[Folder] - [{word_folder}] finished adding padding !")
        print("All padding has been added!")

    def convert_videos_to_numpy(self, input_path, output_path):
        for word_folder in os.listdir(input_path):
            word_folder_path = os.path.join(input_path, word_folder)

            if not os.path.isdir(word_folder_path):
                continue

            word = word_folder

            keypoint_target_path = os.path.join(output_path, word)

            if not os.path.exists(keypoint_target_path):
                os.makedirs(keypoint_target_path)

            for video_file in os.listdir(word_folder_path):
                video_path = os.path.join(word_folder_path, video_file)

                cap = cv2.VideoCapture(video_path)
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                existing_video_folders = [
                    d for d in os.listdir(keypoint_target_path)
                    if os.path.isdir(os.path.join(keypoint_target_path, d))
                ]
                folder_count = len(existing_video_folders) + 1

                sequence_folder_name = f'video_{folder_count}'
                sequence_folder_path = os.path.join(keypoint_target_path, sequence_folder_name)

                if not os.path.exists(sequence_folder_path):
                    os.makedirs(sequence_folder_path)

                if num_frames > 100:
                    step = 2
                else:
                    step = 1

                saved_frame_count = 0
                for frame_num in range(num_frames):
                    ret, image = cap.read()
                    if not ret:
                        break

                    if frame_num % step != 0:
                        continue

                    results, image_res = self.image_processing(image, self.holistic_model)
                    self.draw_landmarks(image_res, results)
                    keypoints = self.keypoint_extraction(results)

                    frame_path = os.path.join(sequence_folder_path, f'w_{word}_s_{folder_count}_f_{saved_frame_count}.npy')
                    np.save(frame_path, keypoints)

                    saved_frame_count += 1

                cap.release()
                print(f"[Word] - [{video_file}] has been converted!")

            print(f"[Folder] - [{word_folder}] has been converted!")

        print("All data has been converted!")

    def get_dataset_vocab(self, input_path):
        keypoints, num_labels = [], []
        word_labels = np.array(os.listdir(input_path))
        map_labels = {label: num for num, label in enumerate(word_labels)}

        for word in word_labels:
            word_folder_path = os.path.join(input_path, word)

            if not os.path.isdir(word_folder_path):
                continue

            for sequence_folder in os.listdir(word_folder_path):
                sequence_folder_path = os.path.join(word_folder_path, sequence_folder)

                if not os.path.isdir(sequence_folder_path):
                    continue

                temp_sequence = []

                for frame_file in sorted(os.listdir(sequence_folder_path), key=lambda x: int(x.split('_f_')[-1].split('.')[0])):
                    if frame_file.endswith('.npy'):
                        frame_path = os.path.join(sequence_folder_path, frame_file)
                        npy = np.load(frame_path)
                        temp_sequence.append(npy)

                if temp_sequence:
                    keypoints.append(temp_sequence)
                    num_labels.append(map_labels[word])

        return word_labels, num_labels, keypoints

    def save_dataset_to_files(self, word_labels, num_labels, keypoints, max_frames, data_type):
        dataset_path = os.path.join(self.files_path, 'dataset.pkl')

        if data_type == "dataset":
           dataset_path = os.path.join(self.files_path, 'dataset.pkl')

        with open(dataset_path, 'wb') as f:
            pickle.dump({
                'word_labels': word_labels,
                'num_labels': num_labels,
                'keypoints': keypoints,
                'max_frames': max_frames
            }, f)

    def load_dataset_from_files(self):
        dataset_path = os.path.join(self.files_path, 'dataset.pkl')
        with open(dataset_path, 'rb') as f:
          data = pickle.load(f)
        word_labels = data['word_labels']
        num_labels = data['num_labels']
        keypoints = data['keypoints']
        max_frames = data['max_frames']
        keypoints, num_labels = np.array(keypoints), to_categorical(num_labels).astype(int)

        return word_labels, max_frames, keypoints, num_labels


