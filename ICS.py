import tensorflow_datasets as tfds
import tensorflow as tf

# Load the dataset with train, validation, and test splits
train_dataset, val_dataset = tfds.load(
    'oxford_iiit_pet',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True  # Returns (image, label) pairs
)
test_dataset = tfds.load(
    'oxford_iiit_pet',
    split='test',
    as_supervised=True
)
#------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------
# Preprocessing function to resize and normalize images
def preprocess(image, label):
    image = tf.image.resize(image, [224, 224])  # Resize to 224x224
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label

# Apply preprocessing to the datasets
train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Batch the datasets
BATCH_SIZE = 32
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Verify the dataset structure
for images, labels in train_dataset.take(1):
    print("Image batch shape:", images.shape)  # Should be (batch_size, 224, 224, 3)
    print("Label batch shape:", labels.shape)  # Should be (batch_size,)


# ----DATA AUGMENTATION---------------------------------------------------------------------------------------------------------

# Apply Data Augmentation 
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# -----LOAD A PRETRAINED MODEL--------------------------------------------------------------
# Create the base model from pre-trained VGG16
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
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(37, activation='softmax')
])

#-------------------------------------------------------------------------------------------

# COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#--------------------------------------------------------------------------------------------
# COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
#-------------------------------------------------------------------------------------------
# EVALUATE THE MODEL
# Preprocess test dataset
test_dataset = test_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}')
#------------------------------------------------------------------------------------------

# PLOT TRAINING HISTORY
import matplotlib.pyplot as plt

# Accuracy plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#--------------------------------------------------------------------------------------------
# Save the model
model.save('vgg16_pet_model.h5')

#--------------------------------------------------------------------------------------------

