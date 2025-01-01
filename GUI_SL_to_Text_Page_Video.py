from GUI_Page import Page
import mediapipe as mp

class SLToTextVideoPage(Page):
    def __init__(self, parent, controller, data_processor):
        super().__init__(parent, controller)
        self.max_camera_width = 800
        self.max_camera_height = 700
        self.running = False
        self.cap = None
        self.holistic_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.data_processor = data_processor

        parent.geometry("1150x700")