import os
import tkinter as tk
import cv2
import mediapipe as mp
import numpy as np
import threading
from GUI_Page import Page
from PIL import Image, ImageTk

class AddDataCameraPage(Page):

    def __init__(self, parent, controller, data_processor):
        super().__init__(parent, controller)

        self.current_frame = None
        self.current_sequence = None
        self.cap = None
        self.running = False
        self.max_camera_width = 800
        self.max_camera_height = 700
        self.camera_current_width = None
        self.camera_current_height = None
        self.word = None
        self.sequences = None
        self.frames = None
        self.video_target_path = None
        self.keypoint_target_path = None
        self.data_processor = data_processor
        self.recording = False
        self.during_recording = False
        self.holistic_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)

        parent.geometry("1150x700")

        # ----- Camera Frame -----
        self.frame_left = tk.Frame(self.frame)
        self.frame_left.grid(row=0, column=0, padx=10, pady=10)

        self.video_label = tk.Label(self.frame_left)
        self.video_label.pack_forget()

        self.placeholder_label = tk.Label(self.frame_left, text="Camera is off", width=100, height=40, bd=2, relief="solid")
        self.placeholder_label.pack()

        # ----- Input Frame -----
        self.frame_right = tk.Frame(self.frame)
        self.frame_right.grid(row=0, column=1, padx=10, pady=10)

        self.label_info_1 = tk.Label(self.frame_right, text="Write word/sign you want to add to DataSet:", font=("Helvetica", 9, 'bold'))
        self.label_info_1.pack()
        self.warning_word = tk.Label(self.frame_right, text="", fg="red")
        self.warning_word.pack()
        self.entry_word = tk.Entry(self.frame_right)
        self.entry_word.pack()

        self.label_info_2 = tk.Label(self.frame_right, text="Define the number of sequences to be recorded for this word/sign:", font=("Helvetica", 9, 'bold'))
        self.label_info_2.pack(pady=(20, 0))
        self.warning_sequences = tk.Label(self.frame_right, text="", fg="red")
        self.warning_sequences.pack()
        self.entry_sequences = tk.Entry(self.frame_right)
        self.entry_sequences.pack()

        self.label_info_3 = tk.Label(self.frame_right, text="Define the number of frames to be recorded for this word/sign:", font=("Helvetica", 9, 'bold'))
        self.label_info_3.pack(pady=(20, 0))
        self.warning_frames = tk.Label(self.frame_right, text="", fg="red")
        self.warning_frames.pack()
        self.entry_frames = tk.Entry(self.frame_right)
        self.entry_frames.pack(pady=(0, 20))

        self.warning_camera = tk.Label(self.frame_right, text="", fg="red")
        self.warning_camera.pack()
        self.button_start = tk.Button(self.frame_right, text="Start Recording", command=self.start_action)
        self.button_start.pack(pady=5)

        self.button_next = tk.Button(self.frame_right, text="Next Sequence", command=self.next_sequence, state=tk.DISABLED)
        self.button_next.pack(pady=5)

        self.button_stop = tk.Button(self.frame_right, text="Stop Recording", command=self.stop_action, state=tk.DISABLED)
        self.button_stop.pack(pady=5)

        # ----- Button Frame -----
        self.frame_bottom = tk.Frame(self.frame)
        self.frame_bottom.grid(row=1, columnspan=2, pady=10)

        self.button_camera = tk.Button(self.frame_bottom, text="Open Camera", command=self.toggle_camera)
        self.button_camera.pack(side=tk.LEFT, padx=5)

        self.button_load_video = tk.Button(self.frame_bottom, text="Load Video", command=self.load_video_action)
        self.button_load_video.pack(side=tk.LEFT, padx=5)

        self.button_home = tk.Button(self.frame_bottom, text="Home", command=self.home_action)
        self.button_home.pack(side=tk.LEFT, padx=5)

    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def stop_camera(self):
        self.button_camera.config(state=tk.DISABLED)
        self.running = False
        if self.cap:
            self.cap.release()
        self.button_camera.config(text="Open Camera", state=tk.NORMAL)
        self.video_label.config(image="", text="")
        self.video_label.pack_forget()
        self.placeholder_label.pack()

    def start_camera(self):
        def camera_thread():
            self.button_camera.config(state=tk.DISABLED)
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.camera_current_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.camera_current_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if self.camera_current_width > self.max_camera_width or self.camera_current_height > self.max_camera_height:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.max_camera_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.max_camera_height)
                self.running = True
                self.update_frame()
            else:
                self.video_label.config(text="No camera detected !", fg="red")
                self.cap.release()
                self.cap = None
                self.button_camera.config(state=tk.NORMAL)

        threading.Thread(target=camera_thread, daemon=True).start()


    def update_frame(self):
        if self.running:
            ret, image = self.cap.read()
            if ret:
                if self.placeholder_label.winfo_ismapped():
                    self.button_camera.config(text="Close Camera", state=tk.NORMAL)
                    self.placeholder_label.pack_forget()
                    self.video_label.pack()
                results, image = self.data_processor.image_processing(image, self.holistic_model)
                self.data_processor.draw_landmarks(image, results)
                if self.recording:
                    if self.during_recording:
                        cv2.circle(image, (self.camera_current_width - 30, 30), 10, (0, 0, 255), -1)
                        cv2.putText(image, f'Recording data for "{self.word}".', (20, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(image, f'Sequence {self.current_sequence}/{self.sequences}',
                                    (20, self.camera_current_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        self.button_next.config(state=tk.DISABLED)
                    else:
                        cv2.putText(image, f'Press Next to Record another Sequence. {self.current_sequence}/{self.sequences} recorded.', (20, self.camera_current_height - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                        self.button_next.config(state=tk.NORMAL)
                else:
                    cv2.putText(image, 'Press Start to Record Sequence', (20, self.camera_current_height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk, text="")
            else:
                if not self.placeholder_label.winfo_ismapped():
                    self.placeholder_label.pack()
            self.video_label.after(5, self.update_frame)
        else:
            self.video_label.config(image="", text="")

    def start_action(self):
        if not self.running:
            self.warning_camera.config(text="You need to turn on your camera first!")
        else:
            self.word = self.entry_word.get().strip()
            self.sequences = self.entry_sequences.get().strip()
            self.frames = self.entry_frames.get().strip()

            self.warning_camera.config(text="")
            self.warning_word.config(text="")
            self.warning_sequences.config(text="")
            self.warning_frames.config(text="")

            valid = True
            if not self.word:
                self.warning_word.config(text="You must enter a word!")
                valid = False
            if not self.sequences.isdigit():
                self.warning_sequences.config(text="You must enter a valid number of sequences!")
                valid = False
            if not self.frames.isdigit():
                self.warning_frames.config(text="You must enter a valid number of frames!")
                valid = False

            if valid:
                if self.word in self.data_processor.list_video_folders_in_directory():
                    self.video_target_path = os.path.join(self.data_processor.dataset_videos_classes, self.word)
                    self.keypoint_target_path = os.path.join(self.data_processor.dataset_keypoints_classes, self.word)
                else:
                    self.video_target_path = os.path.join(self.data_processor.dataset_videos_classes, self.word)
                    os.makedirs(self.video_target_path, exist_ok=True)
                    self.keypoint_target_path = os.path.join(self.data_processor.dataset_keypoints_classes, self.word)
                    os.makedirs(self.keypoint_target_path, exist_ok=True)

                self.current_sequence = 0
                self.current_frame = 0

                self.record_action()
                self.button_stop.config(state=tk.NORMAL)


    def record_data(self):
        def recording_thread():
            if not os.path.exists(self.keypoint_target_path):
                os.makedirs(self.keypoint_target_path)

            num_sequences = int(self.sequences)
            num_frames = int(self.frames)

            if self.current_sequence < num_sequences:
                self.during_recording = True

                existing_video_folders = [d for d in os.listdir(self.keypoint_target_path) if os.path.isdir(os.path.join(self.keypoint_target_path, d))]
                folder_count = len(existing_video_folders) + 1

                sequence_folder_name = f'video_{folder_count}'
                sequence_folder_path = os.path.join(self.keypoint_target_path, sequence_folder_name)

                if not os.path.exists(sequence_folder_path):
                    os.makedirs(sequence_folder_path)

                existing_files = os.listdir(self.video_target_path)
                video_count = len(existing_files)

                existing_keypoint_files = os.listdir(sequence_folder_path)
                keypoint_files_count = len(existing_keypoint_files)

                video_file_name = f'{self.word}_{video_count + 1}.mp4'
                video_file_path = os.path.join(self.video_target_path, video_file_name)

                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                out = cv2.VideoWriter(video_file_path, fourcc, 20.0, (self.camera_current_width, self.camera_current_height))

                for frame_num in range(num_frames):
                    ret, image = self.cap.read()
                    if not ret or not self.recording:
                        break

                    out.write(image)

                    results, image = self.data_processor.image_processing(image, self.holistic_model)
                    self.data_processor.draw_landmarks(image, results)
                    keypoints = self.data_processor.keypoint_extraction(results)

                    frame_path = os.path.join(sequence_folder_path, f'w_{self.word}_{keypoint_files_count + 1}_s_{self.current_sequence}_f_{frame_num}')
                    np.save(frame_path, keypoints)
                    self.current_frame = frame_num

                out.release()
                self.current_sequence += 1

            if self.current_sequence >= num_sequences:
                self.recording = False
                self.during_recording = False
                self.reset_camera_settings()
            else:
                self.during_recording = False
                self.button_next.config(state=tk.NORMAL)

        threading.Thread(target=recording_thread, daemon=True).start()


    def next_sequence(self):
        self.current_frame = 0
        self.record_data()

    def stop_action(self):
        self.recording = False
        self.during_recording = False
        self.reset_camera_settings()
        self.button_stop.config(state=tk.DISABLED)

    def home_action(self):
        self.stop_camera()
        self.stop_action()
        self.controller.show_main_page()

    def load_video_action(self):
        self.stop_camera()
        self.stop_action()
        self.controller.show_add_data_video_page()

    def reset_camera_settings(self):
        self.entry_word.config(state=tk.NORMAL)
        self.entry_sequences.config(state=tk.NORMAL)
        self.entry_frames.config(state=tk.NORMAL)
        self.button_camera.config(state=tk.NORMAL)
        self.button_start.config(state=tk.NORMAL)
        self.button_next.config(state=tk.DISABLED)
        self.button_stop.config(state=tk.NORMAL)

    def record_action(self):
        self.recording = True
        self.entry_word.config(state=tk.DISABLED)
        self.entry_sequences.config(state=tk.DISABLED)
        self.entry_frames.config(state=tk.DISABLED)
        self.button_camera.config(state=tk.DISABLED)
        self.button_start.config(state=tk.DISABLED)
        self.button_next.config(state=tk.NORMAL)
