import tkinter as tk
from tkinter import ttk

class Language_Selector:
    def __init__(self):
        self.language = None
        self._select_language()

    def _select_language(self):
        root = tk.Tk()
        root.title("Select Language")
        root.geometry("400x200")
        root.resizable(False, False)

        lang_var = tk.StringVar()
        languages = ["English", "Polish"]

        frame = tk.Frame(root)
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        label = tk.Label(frame, text="Please select application language:", font=("Helvetica", 12, "bold"))
        label.pack(pady=10)

        lang_dropdown = ttk.Combobox(frame, textvariable=lang_var, values=languages, state="readonly")
        lang_dropdown.current(0)
        lang_dropdown.pack(pady=10)

        def confirm_language():
            self.language = lang_var.get()
            root.destroy()

        confirm_button = tk.Button(frame, text="Confirm", command=confirm_language, bg="#C1C1CD")
        confirm_button.pack(pady=10)

        root.mainloop()
