import matplotlib.pyplot as plt

class Graphs:
    def __init__(self, logs_path):
        self.logs_path = logs_path
        self.create_graphs()

    def create_graphs(self):
        epoch = []
        categorical_accuracy = []
        loss = []
        val_categorical_accuracy = []
        val_loss = []

        with open(self.logs_path, 'r') as file:
            lines = file.readlines()

        for line in lines:

            if 'Epoch' not in line:
                continue

            elements = line.split(';')

            epoch_value = int(elements[0].split(' ')[-1])
            categorical_accuracy_value = float(elements[1].split(':')[-1])
            loss_value = float(elements[2].split(':')[-1])
            val_categorical_accuracy_value = float(elements[3].split(':')[-1])
            val_loss_value = float(elements[4].split(':')[-1])

            epoch.append(epoch_value)
            categorical_accuracy.append(categorical_accuracy_value)
            loss.append(loss_value)
            val_categorical_accuracy.append(val_categorical_accuracy_value)
            val_loss.append(val_loss_value)

        # Graphs
        plt.figure(figsize=(15, 10))

        # Loss / Validation Loss and Epoch graph
        plt.subplot(1, 2, 1)
        plt.plot(epoch, loss, label='Train Loss')
        plt.plot(epoch, val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Validation Loss')
        plt.title('Loss and Validation Loss')
        plt.legend()

        # Categorical Accuracy / Validation Categorical Accuracy and Epoch graph
        plt.subplot(1, 2, 2)
        plt.plot(epoch, categorical_accuracy, label='Train Categorical Accuracy')
        plt.plot(epoch, val_categorical_accuracy, label='Validation Categorical Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Categorical Accuracy / Validation Categorical Accuracy')
        plt.title('Categorical Accuracy and Validation Categorical Accuracy')
        plt.legend()

        # Show graphs
        plt.tight_layout()
        plt.show()