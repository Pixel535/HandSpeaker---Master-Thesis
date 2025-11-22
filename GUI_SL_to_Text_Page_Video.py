import os
import string
import threading
import time
from tkinter import filedialog
import cv2
import language_tool_python
import numpy as np
from keras.src.saving import load_model
from PIL import Image, ImageTk
from GUI_Page import Page
import mediapipe as mp
import tkinter as tk


class SLToTextVideoPage(Page):
    def __init__(self, parent, controller, data_processor, lang):
        super().__init__(parent, controller)
        self.max_video_width = 800
        self.max_video_height = 700
        self.running = False
        self.cap = None
        self.selected_file_path = None
        self.frames = None
        self.holistic_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.data_processor = data_processor
        model_path = os.path.join("Model", lang, "model.keras")
        self.model = load_model(model_path)
        self.tool = None #language_tool_python.LanguageToolPublicAPI('en-UK')
        self.words = self.data_processor.list_video_folders_in_directory()
        self.sentence = []
        self.keypoints_buffer = []
        self.last_prediction = []
        self.grammar_result = ""

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

        self.label_info_1 = tk.Label(self.frame_right, text="Choose a video you want to Translate:", font=("Helvetica", 9, 'bold'))
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

        self.button_start_translating = tk.Button(self.frame_right, text="Start Translating", command=self.start_translating)
        self.button_start_translating.pack(pady=5)

        self.button_grammar_check = tk.Button(self.frame_right, text="Check Grammar", command=self.grammar_check)
        self.button_grammar_check.pack(pady=5)

        self.button_reset = tk.Button(self.frame_right, text="Reset", command=self.reset_translation)
        self.button_reset.pack(pady=5)

        self.label_info_1 = tk.Label(self.frame_right, text="Sign Language translated to Text:", font=("Helvetica", 9, 'bold'))
        self.label_info_1.pack()

        self.label_transcription = tk.Label(self.frame_right, text="", fg="blue", wraplength=300, justify="left")
        self.label_transcription.pack(pady=10)

        # ----- Button Frame -----
        self.frame_bottom = tk.Frame(self.frame)
        self.frame_bottom.grid(row=1, columnspan=2, pady=10)

        self.button_load_video = tk.Button(self.frame_bottom, text="Live Translation", command=self.record_data_action)
        self.button_load_video.pack(side=tk.LEFT, padx=5)

        self.button_home = tk.Button(self.frame_bottom, text="Home", command=self.home_action)
        self.button_home.pack(side=tk.LEFT, padx=5)


    def home_action(self):
        self.stop_video()
        self.controller.show_main_page()

    def record_data_action(self):
        self.stop_video()
        self.controller.show_sl_to_text_camera_page()

    def stop_video(self):
        if self.cap:
            self.cap.release()
        self.selected_file_name.config(text="")
        self.selected_file_path = None
        self.video_label.config(image="", text="")
        self.video_label.pack_forget()
        self.placeholder_label.pack()

    def reset_translation(self):
        self.sentence.clear()
        self.keypoints_buffer.clear()
        self.last_prediction = None
        self.grammar_result = ""
        self.label_transcription.config(text="")

    def grammar_check(self):
        if self.sentence:
            text = ' '.join(self.sentence)
            corrected = self.tool.correct(text)
            self.grammar_result = corrected
        else:
            self.grammar_result = "No text detected"
        self.label_transcription.config(text=self.grammar_result)

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

    def start_translating(self):
        self.warning_video.config(text="")
        valid = True
        if not self.selected_file_path:
            self.warning_video.config(text="You must choose a file!")
            valid = False

        if valid:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.button_choose_file.config(state=tk.DISABLED)
            self.play_video()


    def play_video(self):
        def play_video_thread():
            self.keypoints_buffer.clear()
            if self.cap:
                while True:
                    ret, image = self.cap.read()

                    if ret:
                        if self.placeholder_label.winfo_ismapped():
                            self.placeholder_label.pack_forget()
                            self.video_label.pack()

                        results, image = self.data_processor.image_processing(image, self.holistic_model)
                        self.data_processor.draw_landmarks(image, results)
                        keypoints = self.data_processor.extract_frame_features(results, do_augment=False)
                        self.keypoints_buffer.append(keypoints)

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

                        def update_gui(imgtk_copy):
                            if self.placeholder_label.winfo_ismapped():
                                self.placeholder_label.pack_forget()
                                self.video_label.pack()
                            self.video_label.imgtk = imgtk_copy
                            self.video_label.configure(image=imgtk_copy)

                        self.video_label.after(1, update_gui, imgtk)
                        time.sleep(1 / 10000)
                    else:
                        break

                n_frames = len(self.keypoints_buffer)
                if n_frames == 0:
                    self.button_choose_file.config(state=tk.NORMAL)
                    return
                if n_frames > 100:
                    keypoints_array = self.keypoints_buffer[::2]
                    n_skip = len(keypoints_array)
                    if n_skip > 100:
                        keypoints_array = keypoints_array[:100]
                    elif n_skip < 100:
                        needed = 100 - n_skip
                        zero_frame = np.zeros(190, dtype=float)
                        for _ in range(needed):
                            keypoints_array.append(zero_frame)
                elif n_frames < 100:
                    keypoints_array = self.keypoints_buffer.copy()
                    zero_frame = np.zeros(190, dtype=float)
                    diff = 100 - n_frames
                    for _ in range(diff):
                        keypoints_array.append(zero_frame)
                else:
                    keypoints_array = self.keypoints_buffer.copy()

                kpts = np.array(keypoints_array)
                self.keypoints_buffer.clear()

                prediction = self.model.predict(kpts[np.newaxis, :, :])
                predicted_sign = self.words[np.argmax(prediction)]

                if predicted_sign != self.last_prediction:
                    self.last_prediction = predicted_sign
                    if len(predicted_sign) == 1 and predicted_sign.isalpha():
                        predicted_sign = predicted_sign.upper()
                    self.sentence.append(predicted_sign)

                if len(self.sentence) > 10:
                    self.sentence = self.sentence[-10:]

                if len(self.sentence) >= 2:
                    if all(len(s) == 1 and s.isalpha() and s.isupper() for s in self.sentence[-2:]):
                        i = len(self.sentence) - 1
                        while i > 0 and len(self.sentence[i - 1]) == 1 and self.sentence[i - 1].isupper():
                            i -= 1
                        joined = ''.join(self.sentence[i:]).capitalize()
                        self.sentence = self.sentence[:i] + [joined]

                if self.sentence:
                    self.sentence[0] = self.sentence[0].capitalize()

                text_to_display = self.grammar_result if self.grammar_result else ' '.join(self.sentence)
                self.label_transcription.config(text=text_to_display)
                self.button_choose_file.config(state=tk.NORMAL)

        threading.Thread(target=play_video_thread, daemon=True).start()