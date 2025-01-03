import tkinter as tk
from GUI_AddDataCameraPage import AddDataCameraPage
from GUI_AddDataVideoPage import AddDataVideoPage
from GUI_MainPage import MainPage
from GUI_SL_to_Text_Page_Camera import SLToTextCameraPage
from GUI_SL_to_Text_Page_Video import SLToTextVideoPage
from GUI_Text_to_SL_Page import TextToSLPage


class GUI:
    def __init__(self, data_processor):
        self.window = tk.Tk()
        self.window.title("HandSpeaker")
        self.window.geometry("800x700")
        self.icon = tk.PhotoImage(file="App/Logo.PNG")
        self.window.iconphoto(False, self.icon)
        self.data_processor = data_processor
        self.current_page = None

    def run(self):
        self.show_main_page()
        self.window.mainloop()

    def show_main_page(self):
        self.hide_current_page()
        self.current_page = MainPage(self.window, self)
        self.current_page.show()

    def show_add_data_camera_page(self):
        self.hide_current_page()
        self.current_page = AddDataCameraPage(self.window, self, self.data_processor)
        self.current_page.show()

    def show_add_data_video_page(self):
        self.hide_current_page()
        self.current_page = AddDataVideoPage(self.window, self, self.data_processor)
        self.current_page.show()

    def show_text_to_sl_page(self):
        self.hide_current_page()
        self.current_page = TextToSLPage(self.window, self, self.data_processor)
        self.current_page.show()

    def show_sl_to_text_camera_page(self):
        self.hide_current_page()
        self.current_page = SLToTextCameraPage(self.window, self, self.data_processor)
        self.current_page.show()

    def show_sl_to_text_video_page(self):
        self.hide_current_page()
        self.current_page = SLToTextVideoPage(self.window, self, self.data_processor)
        self.current_page.show()

    def hide_current_page(self):
        if self.current_page:
            self.current_page.hide()
