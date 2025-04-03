
# Image Classification Model

## Overview
This project implements an image classification system using **Transfer Learning** with a **VGG16 Convolutional Neural Network (CNN)**. The model is trained on the **Oxford-IIIT Pet Dataset**, which contains 37 different categories of pet images.

## Features
- **Transfer Learning with VGG16**: Utilizes a pretrained VGG16 model for feature extraction.
- **Data Augmentation**: Includes random horizontal flipping and rotation to improve generalization.
- **Custom Classification Head**: Uses fully connected layers for final predictions.
- **Dropout Regularization**: Helps prevent overfitting by randomly deactivating neurons.

## Installation
To set up the project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/frediking/IMAGE-CLASSIFICATION.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd IMAGE-CLASSIFICATION
   ```
3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Prepare your dataset**: The Oxford-IIIT Pet dataset is automatically downloaded via TensorFlow Datasets.
2. **Train the model**:
   ```bash
   python train.py --epochs 20
   ```
   Adjust the number of epochs as needed.
3. **Evaluate the model**:
   ```bash
   python evaluate.py --model saved_models/model_epoch_20.h5
   ```
   Replace the model path if using a different checkpoint.

## Model Architecture
```python
# Load the pretrained VGG16 model
base_model = tf.keras.applications.VGG16(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
# Freeze the base model
base_model.trainable = False

# Create the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    base_model,  # Pretrained VGG16 CNN
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(37, activation='softmax')
])
```

## Performance Metrics
The model was trained using different configurations:
- **10 epochs**: Initial results.
- **20 epochs**: Improved performance.

Evaluation results, accuracy plots, and confusion matrices can be found in the `reports/` directory.

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add some feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/YourFeatureName
   ```
5. Open a pull request.

6. #### CHECK RELEASES FOR SAVED MODELS 
   #### First model used 10 Epochs
   #### Second model used 20 Epoch

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author
Fredinard Ohene-Addo

