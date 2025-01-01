import threading
import mediapipe as mp
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from GUI_Page import Page


class SLToTextCameraPage(Page):
    def __init__(self, parent, controller, data_processor):
        super().__init__(parent, controller)
        self.max_camera_width = 800
        self.max_camera_height = 700
        self.running = False
        self.translating = False
        self.cap = None
        self.camera_current_width = None
        self.camera_current_height = None
        self.holistic_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.data_processor = data_processor

        parent.geometry("1150x700")

        # ----- Camera Frame -----
        self.frame_left = tk.Frame(self.frame)
        self.frame_left.grid(row=0, column=0, padx=10, pady=10)

        self.video_label = tk.Label(self.frame_left)
        self.video_label.pack_forget()

        self.placeholder_label = tk.Label(self.frame_left, text="Camera is off", width=100, height=40, bd=2, relief="solid")
        self.placeholder_label.pack()

        # ----- Read SL Frame -----
        self.frame_right = tk.Frame(self.frame)
        self.frame_right.grid(row=0, column=1, padx=10, pady=10)

        self.warning_camera = tk.Label(self.frame_right, text="", fg="red")
        self.warning_camera.pack()

        self.button_start_translating = tk.Button(self.frame_right, text="Start Translating", command=self.start_translating)
        self.button_start_translating.pack(pady=5)

        self.button_stop = tk.Button(self.frame_right, text="Stop Translating", command=self.stop_action, state=tk.DISABLED)
        self.button_stop.pack(pady=5)

        self.label_info_1 = tk.Label(self.frame_right, text="Sign Language translated to Text:", font=("Helvetica", 9, 'bold'))
        self.label_info_1.pack()

        # ----- Button Frame -----
        self.frame_bottom = tk.Frame(self.frame)
        self.frame_bottom.grid(row=1, columnspan=2, pady=10)

        self.button_camera = tk.Button(self.frame_bottom, text="Open Camera", command=self.toggle_camera)
        self.button_camera.pack(side=tk.LEFT, padx=5)

        self.button_load_video = tk.Button(self.frame_bottom, text="Load Video", command=self.load_video_action)
        self.button_load_video.pack(side=tk.LEFT, padx=5)

        self.button_home = tk.Button(self.frame_bottom, text="Home", command=self.home_action)
        self.button_home.pack(side=tk.LEFT, padx=5)

    def home_action(self):
        self.stop_camera()
        self.stop_action()
        self.controller.show_main_page()

    def load_video_action(self):
        self.stop_camera()
        self.stop_action()
        self.controller.show_sl_to_text_video_page()

    def stop_camera(self):
        self.button_camera.config(state=tk.DISABLED)
        self.running = False
        if self.cap:
            self.cap.release()
        self.button_camera.config(text="Open Camera", state=tk.NORMAL)
        self.video_label.config(image="", text="")
        self.video_label.pack_forget()
        self.placeholder_label.pack()

    def stop_action(self):
        self.translating = False
        self.reset_camera_settings()
        self.button_stop.config(state=tk.DISABLED)

    def reset_camera_settings(self):
        self.button_camera.config(state=tk.NORMAL)
        self.button_stop.config(state=tk.NORMAL)

    def translate_action(self):
        self.translating = True
        self.button_camera.config(state=tk.DISABLED)
        self.button_start_translating.config(state=tk.DISABLED)

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

    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_translating(self):
        if not self.running:
            self.warning_camera.config(text="You need to turn on your camera first!")
        else:
            self.warning_camera.config(text="")

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
                if self.translating:
                    cv2.putText(image, 'Translating...', (20, self.camera_current_height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, 'Press Start to Translate', (20, self.camera_current_height - 20),
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