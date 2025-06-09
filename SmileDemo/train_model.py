# train_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import os
import sys

# --- Constants ---
# Dataset: https://www.kaggle.com/datasets/chazzer/smiling-or-not-face-data
# Your folder structure should be:
# /dataset
#   /smile
#       /image1.jpg ...
#   /non_smile
#       /image1.jpg ...
#   /test
#       /image1.jpg ... (This folder is for final evaluation)
DATASET_PATH = 'dataset'
TEST_PATH = os.path.join(DATASET_PATH, 'test')
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
MODEL_PATH = 'smile_cnn_model.h5'

def build_model(input_shape):
    """Builds a simple CNN for smile classification."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        Dense(1, activation='sigmoid') # Sigmoid for binary classification
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

def main():
    """Main function to train and save the model."""
    # Check if the main dataset directory exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset directory not found at '{DATASET_PATH}'")
        print("Please download the dataset and ensure it has 'smile' and 'non_smile' subfolders.")
        sys.exit(1)

    # --- Data Augmentation and Generators ---
    # We apply augmentation to the training data to make the model more robust
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # Use 20% of data for validation
    )

    # Note: Keras will look for subdirectories inside DATASET_PATH
    # It will ignore the 'test' directory if we don't explicitly point to it.
    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['non_smile', 'smile'], # Explicitly set class order
        subset='training', # Set as training data
        # Exclude the test directory from training/validation
        exclude_dirs=[TEST_PATH] 
    )

    validation_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['non_smile', 'smile'],
        subset='validation', # Set as validation data
        exclude_dirs=[TEST_PATH]
    )

    # --- Build and Train the Model ---
    model = build_model(input_shape=(*IMAGE_SIZE, 3))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', save_best_only=True)
    ]
    
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=100, # EarlyStopping will stop it when it's done
        callbacks=callbacks
    )

    print(f"\n--- Training Complete ---")
    print(f"Best model saved to '{MODEL_PATH}'")

    # --- Final Evaluation on the Test Set ---
    if os.path.exists(TEST_PATH):
        print("\n--- Evaluating model on the test set ---")
        best_model = load_model(MODEL_PATH)
        
        # Create a new generator for the test data (no augmentation, just rescaling)
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            DATASET_PATH,
            classes=['test'],
            target_size=IMAGE_SIZE,
            batch_size=1, # Process one by one
            shuffle=False,
            class_mode=None # No labels for test set
        )
        
        # Unfortunately, Keras doesn't have a simple way to evaluate a test set
        # without labels. You would typically have a labeled test set for evaluation.
        # This section is left as a placeholder for how you would do it if the test
        # set had 'smile' and 'non_smile' subfolders.
        print("Note: The provided 'test' folder has no labels for a direct accuracy score.")
        print("To get a test accuracy, the test folder should also contain 'smile' and 'non_smile' sub-folders.")

if __name__ == '__main__':
    main()
