import matplotlib.pyplot as plt

def plot_training_accuracy(history):
    """
    Plot the training and validation accuracy over epochs
    """
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_training_loss(history):
    """
    Plot the training and validation loss over epochs
    """
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def visualize_training_history(history):
    """
    Visualize both accuracy and loss plots
    """
    plot_training_accuracy(history)
    plot_training_loss(history)

# Usage example:
# visualize_training_history(history)




