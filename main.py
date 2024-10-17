from Data_Processing import DataProcessing
from GUI import GUI

if __name__ == "__main__":

    data_processor = DataProcessing()

    #logs_path = "Model/logs.log"
    #Graphs(logs_path)

    app = GUI(data_processor)
    app.run()