import os
import mediapipe as mp
import cv2
import numpy as np

class DataProcessing:

    def __init__(self):

        self.dataset_videos_path = "Data/Dataset/SL"
        self.dataset_keypoints_path = "Data/Dataset/Keypoints"
        self.dictionary_path = "Data/Dictionary"

    def get_vocab_and_dict(self):
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

