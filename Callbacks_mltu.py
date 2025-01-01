import os
from keras.src.callbacks import Callback
import logging


class TrainLogger(Callback):

    def __init__(self, log_path: str, log_file: str = 'logs.log', logLevel=logging.INFO) -> None:
        super().__init__()
        self.log_path = log_path
        self.log_file = log_file

        if not os.path.exists(log_path):
            os.mkdir(log_path)

        self.logger = logging.getLogger()
        self.logger.setLevel(logLevel)

        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        self.file_handler = logging.FileHandler(os.path.join(self.log_path, self.log_file))
        self.file_handler.setLevel(logLevel)
        self.file_handler.setFormatter(self.formatter)

        self.logger.addHandler(self.file_handler)

    def on_epoch_end(self, epoch: int, logs: dict = None):
        epoch_message = f"Epoch {epoch}; "
        logs_message = "; ".join([f"{key}: {value}" for key, value in logs.items()])
        self.logger.info(epoch_message + logs_message)