import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Define the directory for testing data
test_dir = os.path.join("data", "natural_images")
model_path = "natural_images_classifier.h5"

# Load the pre-trained model
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file '{model_path}' not found. Train the model before testing.")

model = load_model(model_path)
print(f"Loaded model from {model_path}")

# Function to predict and display an image


def predict_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_labels = ['airplane', 'car', 'cat', 'dog',
                    'flower', 'fruit', 'motorbike', 'person']
    predicted_class = class_labels[np.argmax(prediction)]
    print(f"Predicted class: {predicted_class}")

# Test the model with multiple classes


# Test the model with multiple classes
def test_multiple_classes():
    test_classes = ['airplane', 'car', 'cat', 'dog',
                    'flower', 'fruit', 'motorbike', 'person']
    for test_class in test_classes:
        class_dir = os.path.join(test_dir, test_class)
        random_image = np.random.choice(os.listdir(class_dir))
        sample_image_path = os.path.join(class_dir, random_image)
        print(f"Testing with a random sample from class: {test_class}")
        predict_image(sample_image_path)


# Run the tests
test_multiple_classes()
print("Testing completed.")
