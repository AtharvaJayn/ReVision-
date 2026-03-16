import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# --- CONFIGURATION ---
DATASET_DIR = 'dataset'
IMG_SIZE = (224, 224) 
BATCH_SIZE = 32
EPOCHS = 12  # Slightly increased because the model has to learn from distorted images now

print("--- 1. SETTING UP DATA AUGMENTATION ---")

# 1. Training Generator (WITH Augmentation)
# This artificially rotates, shifts, and flips the images to make the model robust
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,      # Randomly rotate images up to 30 degrees
    width_shift_range=0.2,  # Randomly shift images left/right
    height_shift_range=0.2, # Randomly shift images up/down
    shear_range=0.2,        # Slightly distort the image
    zoom_range=0.2,         # Randomly zoom in/out
    horizontal_flip=True,   # Randomly flip images (mirror effect)
    validation_split=0.2    
)

# 2. Validation Generator (NO Augmentation)
# We test on normal, undistorted images to get an accurate accuracy score
test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = test_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print(f"\nClasses found: {train_generator.class_indices}")

# --- MODEL ARCHITECTURE ---
print("\n--- 2. BUILDING THE MODEL (MobileNetV2) ---")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
# ADDED: Dropout layer. This randomly turns off 20% of neurons to prevent "overfitting" (memorizing the data).
x = Dropout(0.2)(x) 
x = Dense(128, activation='relu')(x) 
predictions = Dense(6, activation='softmax')(x) 

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# --- TRAINING ---
print("\n--- 3. STARTING TRAINING (With Augmented Data) ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# --- SAVING ---
print("\n--- 4. SAVING THE SMARTER MODEL ---")
model.save('eco_sorter.h5')
print("✅ Success! Smarter model saved as 'eco_sorter.h5'.")