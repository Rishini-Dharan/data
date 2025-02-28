import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define Image Data Paths
TRAIN_DIR = "data/ore_images/train"
VAL_DIR = "data/ore_images/val"

# Image Data Generator
datagen = ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory(TRAIN_DIR, target_size=(150,150), batch_size=32, class_mode='categorical')
val_data = datagen.flow_from_directory(VAL_DIR, target_size=(150,150), batch_size=32, class_mode='categorical')

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: Low, Medium, High
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_data, validation_data=val_data, epochs=10)

# Save Model
model.save("models/ore_classifier.h5")
print("✅ Ore Sorting Model Saved!")
