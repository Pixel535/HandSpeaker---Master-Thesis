from Data_Processing import DataProcessing
from GUI import GUI

if __name__ == "__main__":

    convert_data = False
    data_processor = DataProcessing()

    if convert_data:
        data_processor.convert_videos_to_numpy()
    else:

        print(data_processor.get_dataset_vocab())
        #logs_path = "Model/logs.log"
        #Graphs(logs_path)

        app = GUI(data_processor)
        app.run()