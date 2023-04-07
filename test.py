from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


# Load the saved classifier model
model_path = "mnist_classifier.h5"
loaded_classifier = load_model(model_path)
print("Model loaded from", model_path)


def predict_single_image(image_array, classifier):
    # Preprocess the image
    image = np.expand_dims(image_array, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = image.astype('float32') / 255.0

    # Predict the digit using the trained classifier
    prediction = classifier.predict(image)
    digit = np.argmax(prediction)

    return digit




# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Select an image from the test set
image_index = 0
image_to_predict = x_test[image_index]


# Use the predict_single_image function
predicted_digit = predict_single_image(image_to_predict, loaded_classifier)
print(f"The predicted digit for the input image at index {image_index} is: {predicted_digit}")
print(f"The true label for the input image at index {image_index} is: {y_test[image_index]}")



