import os
import cv2
import pickle
import shutil
import pandas as pd
import numpy as np
import mediapipe as mp
from keras.src.utils import to_categorical


class DataProcessing:

    def __init__(self, lang):
        self.dictionary_path = "Data/Dictionary"

        base_lang_path = os.path.join("Data", lang)

        self.dataset_videos_classes = os.path.join(base_lang_path, "Dataset", "Videos")
        self.dataset_keypoints_classes = os.path.join(base_lang_path, "Dataset", "Keypoints")
        self.dataset_videos_rejected = os.path.join(base_lang_path, "Dataset", "Rejected Videos")
        self.dataset_downloaded_videos = os.path.join(base_lang_path, "Dataset", "Downloaded Videos")

        self.files_path = os.path.join(base_lang_path, "Files")

        self.dataset_test_file_path = os.path.join(base_lang_path, "Dataset", "test.csv")
        self.dataset_train_file_path = os.path.join(base_lang_path, "Dataset", "train.csv")
        self.dataset_val_file_path = os.path.join(base_lang_path, "Dataset", "val.csv")

        self.holistic_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mask_value = 0.0
        self.PER_HAND_DIM = 95
        self.FRAME_DIM = self.PER_HAND_DIM * 2

        # =========================
        #  Indeksy MediaPipe: Hands
        # =========================
        self.WRIST = 0
        self.THUMB_CMC, self.THUMB_MCP, self.THUMB_IP, self.THUMB_TIP = 1, 2, 3, 4
        self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP = 5, 6, 7, 8
        self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_DIP, self.MIDDLE_TIP = 9, 10, 11, 12
        self.RING_MCP, self.RING_PIP, self.RING_DIP, self.RING_TIP = 13, 14, 15, 16
        self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP = 17, 18, 19, 20

        self.FINGERS = [
            (self.THUMB_CMC, self.THUMB_MCP, self.THUMB_IP, self.THUMB_TIP),
            (self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP),
            (self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_DIP, self.MIDDLE_TIP),
            (self.RING_MCP, self.RING_PIP, self.RING_DIP, self.RING_TIP),
            (self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP),
        ]

        # =========================
        #  Indeksy MediaPipe: Pose (głowa)
        # =========================
        self.POSE_NOSE = 0
        self.POSE_LEFT_EYE = 2
        self.POSE_RIGHT_EYE = 5
        self.POSE_MOUTH_LEFT = 9
        self.POSE_MOUTH_RIGHT = 10

    def list_video_folders_in_directory(self):
        return [name for name in os.listdir(self.dataset_videos_classes)
                if os.path.isdir(os.path.join(self.dataset_videos_classes, name))]

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

    def safe_norm(self, v, axis=-1, keepdims=False, eps=1e-6):
        return np.linalg.norm(v, axis=axis, keepdims=keepdims) + eps

    def normalize_hand_xy_with_params(self, points_xy, handedness):
        pts = points_xy.astype(np.float32).copy()
        wrist = pts[self.WRIST:self.WRIST + 1, :].copy()
        pts -= wrist

        palm_vec = pts[self.MIDDLE_MCP, :]
        scale = self.safe_norm(palm_vec)
        pts /= scale

        v = pts[self.INDEX_MCP, :]
        ang = -np.arctan2(v[1], v[0])
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts = pts @ R.T

        mirrored = False
        if handedness == 'left':
            pts[:, 0] = -pts[:, 0]
            mirrored = True

        return pts, (wrist.reshape(2), float(scale), R, mirrored)

    def finger_joint_angles(self, points_xy_norm):
        def angle(a, b, c):
            ba = a - b
            bc = c - b
            cosang = np.dot(ba, bc) / (self.safe_norm(ba) * self.safe_norm(bc))
            cosang = np.clip(cosang, -1.0, 1.0)
            return np.arccos(cosang)

        angs = []
        angs.append(angle(points_xy_norm[self.THUMB_CMC], points_xy_norm[self.THUMB_MCP], points_xy_norm[self.THUMB_IP]))
        angs.append(angle(points_xy_norm[self.THUMB_MCP], points_xy_norm[self.THUMB_IP], points_xy_norm[self.THUMB_TIP]))

        for mcp, pip, dip, tip in self.FINGERS[1:]:
            angs.append(angle(points_xy_norm[mcp], points_xy_norm[pip], points_xy_norm[dip]))
            angs.append(angle(points_xy_norm[pip], points_xy_norm[dip], points_xy_norm[tip]))
        return np.array(angs, dtype=np.float32)

    def bone_vectors(self, points_xy_norm):
        segs = []
        for a, b, c, d in self.FINGERS:
            segs += [
                points_xy_norm[b] - points_xy_norm[a],
                points_xy_norm[c] - points_xy_norm[b],
                points_xy_norm[d] - points_xy_norm[c],
            ]
        for mcp in [self.INDEX_MCP, self.MIDDLE_MCP, self.RING_MCP, self.PINKY_MCP, self.THUMB_MCP]:
            segs.append(points_xy_norm[mcp] - points_xy_norm[self.WRIST])

        return np.concatenate(segs, axis=0).astype(np.float32)

    def pose_head_refs(self, pose_points_xy):
        if pose_points_xy is None:
            return None
        return {
            'nose': pose_points_xy[self.POSE_NOSE, :2].astype(np.float32),
            'eyeL': pose_points_xy[self.POSE_LEFT_EYE, :2].astype(np.float32),
            'eyeR': pose_points_xy[self.POSE_RIGHT_EYE, :2].astype(np.float32),
            'mouthL': pose_points_xy[self.POSE_MOUTH_LEFT, :2].astype(np.float32),
            'mouthR': pose_points_xy[self.POSE_MOUTH_RIGHT, :2].astype(np.float32),
        }

    def hand_to_head_distances(self, pose_head_xy_raw, wrist_scale_ref):
        if pose_head_xy_raw is None or wrist_scale_ref is None:
            return np.zeros((3,), dtype=np.float32)

        wrist_raw, scale_scalar, R, mirrored = wrist_scale_ref

        def norm_point_head(p):
            p = p.astype(np.float32).copy()
            p = p - wrist_raw
            p = p / scale_scalar
            p = (R @ p.reshape(2, 1)).reshape(2)
            if mirrored:
                p[0] = -p[0]
            return p

        head_norm = {k: norm_point_head(v) for k, v in pose_head_xy_raw.items()}
        centers = [
            head_norm['nose'],
            0.5 * (head_norm['eyeL'] + head_norm['eyeR']),
            0.5 * (head_norm['mouthL'] + head_norm['mouthR']),
        ]
        dists = [self.safe_norm(c) for c in centers]
        return np.array(dists, dtype=np.float32)

    def build_hand_features_2D(self, hand_xy_raw, handedness, pose_xy_raw):
        PER_HAND_DIM = 95
        if hand_xy_raw is None:
            return np.zeros((PER_HAND_DIM,), dtype=np.float32)

        xy_norm, params = self.normalize_hand_xy_with_params(hand_xy_raw, handedness=handedness)
        angs = self.finger_joint_angles(xy_norm)
        bones = self.bone_vectors(xy_norm)
        head = self.pose_head_refs(pose_xy_raw) if pose_xy_raw is not None else None
        pose_d = self.hand_to_head_distances(head, wrist_scale_ref=params)

        feats = np.concatenate([
            xy_norm.reshape(-1),
            bones,
            angs,
            pose_d
        ]).astype(np.float32)
        return feats

    def augment_hand_xy(self, points_xy, p_rot=0.7, p_scale=0.7, p_shift=0.7, p_jitter=0.7):
        pts = points_xy.astype(np.float32).copy()

        if np.random.rand() < p_rot:
            ang = np.deg2rad(np.random.uniform(-35, 35))
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            pts = pts @ R.T

        if np.random.rand() < p_scale:
            sc = np.random.uniform(0.85, 1.15)
            pts *= sc

        if np.random.rand() < p_shift:
            dx, dy = np.random.uniform(-0.1, 0.1, size=2)
            pts += np.array([dx, dy], dtype=np.float32)

        if np.random.rand() < p_jitter:
            noise = np.random.normal(0.0, 0.01, size=pts.shape).astype(np.float32)
            pts += noise

        return pts


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
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)


    def _results_to_arrays(self, results):
        left_xy = None
        right_xy = None
        pose_xy = None

        if results.left_hand_landmarks:
            left_xy = np.array([[lm.x, lm.y] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
        if results.right_hand_landmarks:
            right_xy = np.array([[lm.x, lm.y] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
        if results.pose_landmarks:
            pose_xy = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark], dtype=np.float32)

        return left_xy, right_xy, pose_xy

    def extract_frame_features(self, results, do_augment=True):
        left_xy, right_xy, pose_xy = self._results_to_arrays(results)

        if do_augment:
            if left_xy is not None:
                left_xy = self.augment_hand_xy(left_xy)
            if right_xy is not None:
                right_xy = self.augment_hand_xy(right_xy)

        left_feats = self.build_hand_features_2D(left_xy, handedness='left', pose_xy_raw=pose_xy)
        right_feats = self.build_hand_features_2D(right_xy, handedness='right', pose_xy_raw=pose_xy)

        return np.concatenate([left_feats, right_feats], axis=0)  # (190,)


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
            if not os.path.isdir(word_folder_path):
                continue

            word = word_folder
            for videoFolder in os.listdir(word_folder_path):
                video_folder_path = os.path.join(word_folder_path, videoFolder)
                if not os.path.isdir(video_folder_path):
                    continue

                files = [file for file in os.listdir(video_folder_path) if file.endswith('.npy')]
                current_file_count = len(files)

                files_to_add = max_files_count - current_file_count
                sequence_folder_name = os.path.basename(video_folder_path)
                folder_count = int(sequence_folder_name.split('_')[1])
                start_frame_num = current_file_count

                for i in range(files_to_add):
                    dummy_data = np.full((self.FRAME_DIM,), self.mask_value, dtype=np.float32)
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
                    feats = self.extract_frame_features(results, do_augment=True)

                    frame_path = os.path.join(sequence_folder_path, f'w_{word}_s_{folder_count}_f_{saved_frame_count}.npy')
                    np.save(frame_path, feats)
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
                frame_files = [f for f in os.listdir(sequence_folder_path) if f.endswith('.npy')]
                frame_files_sorted = sorted(
                    frame_files, key=lambda x: int(x.split('_f_')[-1].split('.')[0])
                )

                for frame_file in frame_files_sorted:
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
                'max_frames': max_frames,
                'frame_dim': self.FRAME_DIM,
                'per_hand_dim': self.PER_HAND_DIM
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
