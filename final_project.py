import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Step 2: Data Preprocessing (Related to Homework 2: Data augmentation and preprocessing techniques)
train_dir = os.path.join("data", "natural_images")

# Image augmentation and preprocessing
data_gen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = data_gen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = data_gen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Step 3: Define Neural Network Model (Related to Homework 3: Building and customizing neural network models)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 4: Train the Model (Related to Homework 3: Training models and tracking performance metrics)
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    # epochs=10
    epochs=20
)

# Step 5: Evaluate the Model (Related to Homework 3: Evaluating trained models)
eval_loss, eval_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {eval_loss}")
print(f"Validation Accuracy: {eval_accuracy}")

# Step 6: Save the Model (Related to Homework 1: Saving and loading models)
model.save("natural_images_classifier.h5")
print("Model saved as natural_images_classifier.h5")

# Step 7: Demonstrate Model Usage (Related to Homework 2: Working with predictions and real-world inference)


def predict_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_labels = list(train_generator.class_indices.keys())
    predicted_class = class_labels[np.argmax(prediction)]
    print(f"Predicted class: {predicted_class}")


# Example usage of the model
sample_image_path = os.path.join(train_dir, 'airplane', os.listdir(
    os.path.join(train_dir, 'airplane'))[0])
predict_image(sample_image_path)
print("Natural Images Classifier Demo Completed")
