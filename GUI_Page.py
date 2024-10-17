import tkinter as tk

class Page:
    def __init__(self, parent, controller):
        self.parent = parent
        self.controller = controller
        self.frame = tk.Frame(self.parent)

    def show(self):
        self.frame.pack()
        self.frame.tkraise()

    def hide(self):
        self.frame.pack_forget()