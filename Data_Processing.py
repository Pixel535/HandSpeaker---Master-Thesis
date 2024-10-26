import os
import pickle

import mediapipe as mp
import cv2
import numpy as np

class DataProcessing:

    def __init__(self):

        self.dataset_videos_path = "Data/Dataset/SL"
        self.dataset_keypoints_path = "Data/Dataset/Keypoints"
        self.dictionary_path = "Data/Dictionary"
        self.files_path = "Data/Files"
        self.holistic_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)

    def get_vocab_and_dict_for_Text_to_SL(self):
        vocab = []
        file_dict = {}
        for file in os.listdir(self.dictionary_path):
            file_path = os.path.join(self.dictionary_path, file)
            file_name, file_extension = os.path.splitext(file)
            vocab.append(file_name)
            file_dict[file_name] = file_path
        return vocab, file_dict

    def list_video_folders_in_directory(self):
        folders = [name for name in os.listdir(self.dataset_videos_path) if os.path.isdir(os.path.join(self.dataset_videos_path, name))]
        return folders

    def list_video_folder_paths_in_directory(self):
        folder_paths = [os.path.join(self.dataset_videos_path, name) for name in os.listdir(self.dataset_videos_path) if os.path.isdir(os.path.join(self.dataset_videos_path, name))]
        return folder_paths

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

    def convert_videos_to_numpy(self):
        for word_folder in os.listdir(self.dataset_videos_path):
            word_folder_path = os.path.join(self.dataset_videos_path, word_folder)

            if not os.path.isdir(word_folder_path):
                continue

            word = word_folder

            keypoint_target_path = os.path.join(self.dataset_keypoints_path, word)

            if not os.path.exists(keypoint_target_path):
                os.makedirs(keypoint_target_path)

            for video_file in os.listdir(word_folder_path):
                video_path = os.path.join(word_folder_path, video_file)

                cap = cv2.VideoCapture(video_path)

                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                existing_video_folders = [d for d in os.listdir(keypoint_target_path) if
                                          os.path.isdir(os.path.join(keypoint_target_path, d))]
                folder_count = len(existing_video_folders) + 1

                sequence_folder_name = f'video_{folder_count}'
                sequence_folder_path = os.path.join(keypoint_target_path, sequence_folder_name)

                if not os.path.exists(sequence_folder_path):
                    os.makedirs(sequence_folder_path)

                for frame_num in range(num_frames):
                    ret, image = cap.read()
                    if not ret:
                        break

                    results, image_res = self.image_processing(image, self.holistic_model)
                    self.draw_landmarks(image_res, results)
                    keypoints = self.keypoint_extraction(results)

                    frame_path = os.path.join(sequence_folder_path, f'w_{word}_s_{folder_count}_f_{frame_num}.npy')
                    np.save(frame_path, keypoints)

                cap.release()
                print(f"[Word] - [{video_file}] has been converted!")

            print(f"[Folder] - [{word_folder}] has been converted!")

        print("All data has been converted!")

    def get_dataset_vocab(self):
        keypoints, num_labels = [], []
        word_labels = np.array(os.listdir(self.dataset_keypoints_path))
        map_labels = {label: num for num, label in enumerate(word_labels)}

        for word in word_labels:
            word_folder_path = os.path.join(self.dataset_keypoints_path, word)

            if not os.path.isdir(word_folder_path):
                continue

            for sequence_folder in os.listdir(word_folder_path):
                sequence_folder_path = os.path.join(word_folder_path, sequence_folder)

                if not os.path.isdir(sequence_folder_path):
                    continue

                temp_sequence = []

                for frame_file in sorted(os.listdir(sequence_folder_path)):
                    if frame_file.endswith('.npy'):
                        frame_path = os.path.join(sequence_folder_path, frame_file)
                        npy = np.load(frame_path)
                        temp_sequence.append(npy)

                if temp_sequence:
                    keypoints.append(temp_sequence)
                    num_labels.append(map_labels[word])

        return word_labels, num_labels, keypoints

    def save_dataset_to_files(self, word_labels, num_labels, keypoints):
        dataset_path = os.path.join(self.files_path, 'dataset.pkl')
        with open(dataset_path, 'wb') as f:
            pickle.dump({
                'word_labels': word_labels,
                'num_labels': num_labels,
                'keypoints': keypoints
            }, f)

    def load_dataset_from_files(self):
        dataset_path = os.path.join(self.files_path, 'dataset.pkl')
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)

        word_labels = data['word_labels']
        num_labels = data['num_labels']
        keypoints = data['keypoints']

        return word_labels, num_labels, keypoints


