import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 10

# Load dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/train',
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Extract exact class names
class_names = train_ds.class_names
print("Exact Dataset Classes:", class_names)

# Save exact class names list so app.py uses the SAME order
np.save('class_names.npy', class_names)

# Build CNN
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Retrain model with saved classes
model.fit(train_ds, epochs=EPOCHS)

# Save as lightweight native Keras format
model.save("model.h5")
print("Model and Class Names Saved Successfully as model_light.keras!")