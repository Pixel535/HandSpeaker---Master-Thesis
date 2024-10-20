import os
import tkinter as tk
import cv2
import mediapipe as mp
import numpy as np
import threading
from GUI_Page import Page
from PIL import Image, ImageTk
from tkinter import filedialog

class AddDataVideoPage(Page):

    def __init__(self, parent, controller, data_processor):
        super().__init__(parent, controller)

        self.keypoint_target_path = None
        self.selected_file_path = None
        self.max_video_width = 800
        self.max_video_height = 700
        self.running = False
        self.cap = None
        self.sequences = 1
        self.frames = None
        self.holistic_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.data_processor = data_processor
        self.word = None

        parent.geometry("1150x700")

        # ----- Video Frame -----
        self.frame_left = tk.Frame(self.frame)
        self.frame_left.grid(row=0, column=0, padx=10, pady=10)

        self.video_label = tk.Label(self.frame_left)
        self.video_label.pack_forget()

        self.placeholder_label = tk.Label(self.frame_left, text="Video wasn't loaded", width=100, height=40, bd=2, relief="solid")
        self.placeholder_label.pack()

        # ----- Input Frame -----
        self.frame_right = tk.Frame(self.frame)
        self.frame_right.grid(row=0, column=1, padx=10, pady=10)

        self.label_info_1 = tk.Label(self.frame_right, text="Write word/sign you want to add to DataSet:",font=("Helvetica", 9, 'bold'))
        self.label_info_1.pack()
        self.warning_word = tk.Label(self.frame_right, text="", fg="red")
        self.warning_word.pack()
        self.entry_word = tk.Entry(self.frame_right)
        self.entry_word.pack()

        self.label_info_1 = tk.Label(self.frame_right, text="Choose a video you want to load into the Dataset:", font=("Helvetica", 9, 'bold'))
        self.label_info_1.pack(pady=(20, 10))

        self.TextFrame = tk.Frame(self.frame_right)
        self.TextFrame.pack(pady=5)
        self.selected_file_label = tk.Label(self.TextFrame, text="Chosen File: ", font=("Helvetica", 10))
        self.selected_file_label.pack(side="left", pady=5)
        self.selected_file_name = tk.Label(self.TextFrame, text="", font=("Helvetica", 10, 'bold'))
        self.selected_file_name.pack(side="left", pady=5)

        self.warning_video = tk.Label(self.frame_right, text="", fg="red")
        self.warning_video.pack()

        self.button_choose_file = tk.Button(self.frame_right, text="Choose File", command=self.choose_file_action)
        self.button_choose_file.pack(pady=5)

        self.button_add_to_dataset = tk.Button(self.frame_right, text="Add to Dataset", command=self.add_to_dataset_action)
        self.button_add_to_dataset.pack(pady=5)

        # ----- Button Frame -----
        self.frame_bottom = tk.Frame(self.frame)
        self.frame_bottom.grid(row=1, columnspan=2, pady=10)

        self.button_load_video = tk.Button(self.frame_bottom, text="Record Data", command=self.record_data_action)
        self.button_load_video.pack(side=tk.LEFT, padx=5)

        self.button_home = tk.Button(self.frame_bottom, text="Home", command=self.home_action)
        self.button_home.pack(side=tk.LEFT, padx=5)

    def home_action(self):
        self.stop_video()
        self.controller.show_main_page()

    def record_data_action(self):
        self.stop_video()
        self.controller.show_add_data_page()

    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.selected_file_name.config(text="")
        self.selected_file_path = None
        self.video_label.config(image="", text="")
        self.video_label.pack_forget()
        self.placeholder_label.pack()

    def choose_file_action(self):
        self.stop_video()
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.selected_file_path = file_path
            self.selected_file_name.config(text=os.path.basename(file_path))
            def load_file_thread():
                self.load_video(file_path)
            threading.Thread(target=load_file_thread, daemon=True).start()

    def load_video(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.cap.isOpened():
            self.warning_video.config(text="Error loading video file!")
            return

        ret, image = self.cap.read()
        if ret:
            if self.placeholder_label.winfo_ismapped():
                self.placeholder_label.pack_forget()
                self.video_label.pack()

            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk, text="")

    def add_to_dataset_action(self):

        self.word = self.entry_word.get().strip()

        self.warning_video.config(text="")
        self.warning_word.config(text="")

        valid = True
        if not self.word:
            self.warning_word.config(text="You must enter a word!")
            valid = False
        if not self.selected_file_path:
            self.warning_video.config(text="You must choose a file!")
            valid = False

        if valid:
            if self.word in self.data_processor.list_video_folders_in_directory():
                self.keypoint_target_path = os.path.join(self.data_processor.dataset_keypoints_path, self.word)
            else:
                self.keypoint_target_path = os.path.join(self.data_processor.dataset_keypoints_path, self.word)
                os.makedirs(self.keypoint_target_path, exist_ok=True)

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.button_choose_file.config(state=tk.DISABLED)
            self.play_video()
            self.save_keypoints()



    def save_keypoints(self):
        def recording_thread():
            if not os.path.exists(self.keypoint_target_path):
                os.makedirs(self.keypoint_target_path)

            num_frames = int(self.frames)

            existing_keypoint_files = os.listdir(self.keypoint_target_path)
            keypoint_files_count = len(existing_keypoint_files)

            for frame_num in range(num_frames):
                ret, image = self.cap.read()
                if not ret or not self.running:
                    break

                results, image_res = self.data_processor.image_processing(image, self.holistic_model)
                self.data_processor.draw_landmarks(image_res, results)
                keypoints = self.data_processor.keypoint_extraction(results)

                frame_path = os.path.join(self.keypoint_target_path, f'w_{self.word}_{keypoint_files_count + 1}_s_1_f_{frame_num}')
                print(self.keypoint_target_path, frame_path)
                np.save(frame_path, keypoints)

            self.placeholder_label.config(text="Your data has been added to Dataset! Load another video!")
            self.stop_video()
            self.button_choose_file.config(state=tk.NORMAL)


        threading.Thread(target=recording_thread, daemon=True).start()

    def play_video(self):
        def play_video_thread():
            if self.cap:
                self.running = True

                while self.running:
                    ret, image = self.cap.read()

                    if ret:

                        if self.placeholder_label.winfo_ismapped():
                            self.placeholder_label.pack_forget()
                            self.video_label.pack()

                        results, image = self.data_processor.image_processing(image, self.holistic_model)
                        self.data_processor.draw_landmarks(image, results)

                        height, width, _ = image.shape
                        if width > self.max_video_width or height > self.max_video_height:
                            scale_width = self.max_video_width / width
                            scale_height = self.max_video_height / height
                            scale = min(scale_width, scale_height)

                            new_width = int(width * scale)
                            new_height = int(height * scale)

                            image = cv2.resize(image, (new_width, new_height))

                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(image)
                        imgtk = ImageTk.PhotoImage(image=img)
                        self.video_label.imgtk = imgtk
                        self.video_label.configure(image=imgtk, text="")

                        self.video_label.after(5, lambda: None)
                    else:
                        self.running = False
            self.running = False

        threading.Thread(target=play_video_thread, daemon=True).start()
