import tkinter as tk
from GUI_Page import Page
from PIL import Image, ImageTk

class MainPage(Page):

    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.parent = parent
        self.image_path = "App/Logo.PNG"
        parent.geometry("800x700")

        # ----- Info Labels And Logo -----
        self.label = tk.Label(self.frame, text="Welcome to HandSpeaker - Sign Language Translator !", font=("Helvetica", 20, "bold"))
        self.label.pack(pady=20)
        self.logo = Image.open(self.image_path)
        self.logo_resized = self.logo.resize((300, 300))
        self.logo_tk = ImageTk.PhotoImage(self.logo_resized)
        self.logo_label = tk.Label(self.frame, image=self.logo_tk)
        self.logo_label.pack(pady=20)
        self.info = tk.Label(self.frame, text="Select one of the options: ", font=("Helvetica", 10, "bold"))
        self.info.pack(pady=20)

        # ----- Button 1 -----
        self.button_1 = tk.Button(self.frame, text="Add data to Dataset", bg="#C1C1CD", command=self.Add_Data)
        self.button_1.pack(pady=10)

        # ----- Button 2 -----
        self.button_3 = tk.Button(self.frame, text="Translate SL to Text", bg="#C1C1CD", command=self.SL_to_Text)
        self.button_3.pack(pady=10)

        # ----- Button 3 -----
        self.button_4 = tk.Button(self.frame, text="Translate Text to SL", bg="#C1C1CD", command=self.Text_to_SL)
        self.button_4.pack(pady=10)

    def Add_Data(self):
        self.controller.show_add_data_page()

    def SL_to_Text(self):
        self.controller.show_sl_to_text_page()

    def Text_to_SL(self):
        self.controller.show_text_to_sl_page()