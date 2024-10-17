import tkinter as tk
from PIL import Image, ImageTk
from GUI_Page import Page


class TextToSLPage(Page):

    def __init__(self, parent, controller, data_processor):
        super().__init__(parent, controller)
        self.row_frame = None
        self.image_label = None
        self.image_frame = None
        self.word_label = None
        self.word_frame = None
        self.data_processor = data_processor
        self.vocab, self.file_dict = self.data_processor.get_vocab_and_dict()

        parent.geometry("1150x700")

        # ----- Translate Frame -----
        self.left_width = 1150 * (2 / 3) - 100
        self.max_row_width = self.left_width - 100
        self.frame_left = tk.Frame(self.frame, width=int(self.left_width), height=700)
        self.frame_left.grid(row=0, column=0, padx=10, pady=10)

        self.canvas = tk.Canvas(self.frame_left, width=int(self.left_width), height=700)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = tk.Scrollbar(self.frame_left, command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.translate_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.translate_frame, anchor='nw')

        self.translate_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind_all("<Button-4>", self.on_mousewheel)
        self.canvas.bind_all("<Button-5>", self.on_mousewheel)

        # ----- Input Frame -----
        self.right_width = 1150 * (1 / 3)
        self.frame_right = tk.Frame(self.frame, width=int(self.right_width), height=700)
        self.frame_right.grid(row=0, column=1, padx=10, pady=10)

        self.label = tk.Label(self.frame_right, text="Write your text to translate it to Sign Language.", font=("Arial", 14))
        self.label.pack(pady=(10, 10))

        self.textbox = tk.Text(self.frame_right, width=40, height=10, font=("Arial", 12))
        self.textbox.pack(pady=(0, 20))

        self.error_label = tk.Label(self.frame_right, text="", fg="red", font=("Arial", 12))
        self.error_label.pack(pady=(5, 15))

        self.button_frame = tk.Frame(self.frame_right)
        self.button_frame.pack()

        self.translate_button = tk.Button(self.button_frame, text="Translate", command=self.translate_text)
        self.translate_button.pack(side=tk.LEFT, padx=(0, 10))

        self.home_button = tk.Button(self.button_frame, text="Home", command=self.home_action)
        self.home_button.pack(side=tk.LEFT)

    def translate_text(self):
        input_text = self.textbox.get("1.0", tk.END).strip().lower()
        self.error_label.config(text="")

        for char in input_text:
            if char != " " and char.upper() not in self.file_dict:
                self.error_label.config(text=f"Error: The character '{char}' is not in the dictionary.")
                return

        for widget in self.translate_frame.winfo_children():
            widget.destroy()

        words = input_text.split()
        word_spacing = 10
        current_row_width = 0
        self.row_frame = tk.Frame(self.translate_frame)
        self.row_frame.pack(anchor="w", fill=tk.X, pady=10)

        for word in words:
            total_word_width = 0
            letter_images = []

            for letter in word:
                image_path = self.file_dict.get(letter.upper())
                if image_path:
                    img = Image.open(image_path)
                    img = img.resize((75, 75))
                    letter_images.append(img)
                    total_word_width += 75

            if total_word_width > self.max_row_width:
                base_font_size = 10
                max_length_for_default_font = 15
                font_size = max(base_font_size - int((len(word) - max_length_for_default_font) / 2), 6)

                self.error_label.config(text=f"Error: The word '{word}' is too long to fit in the frame.", font=("Arial", font_size))
                return

            if current_row_width + total_word_width + word_spacing > self.max_row_width:
                current_row_width = 0
                self.row_frame = tk.Frame(self.translate_frame)
                self.row_frame.pack(anchor="w", fill=tk.X, pady=10)

            self.word_frame = tk.Frame(self.row_frame)
            self.word_frame.pack(side=tk.LEFT, padx=word_spacing)

            self.word_label = tk.Label(self.word_frame, text=word, font=("Arial", 14), anchor="w")
            self.word_label.pack(side=tk.TOP, anchor="w", fill=tk.X)

            self.image_frame = tk.Frame(self.word_frame)
            self.image_frame.pack(anchor="w")

            for img in letter_images:
                letter_image = ImageTk.PhotoImage(img)
                self.image_label = tk.Label(self.image_frame, image=letter_image)
                self.image_label.image = letter_image
                self.image_label.pack(side=tk.LEFT)
            current_row_width += total_word_width + word_spacing

    def on_mousewheel(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def home_action(self):
        self.controller.show_main_page()