import json,keras 
import numpy as np
import pandas as pd
from keras.preprocessing import image as image_utils
import io
import config

class Rice_Type_Clf():

    def __init__(self):
        self.model = None
        self.class_names = None

    def load_model(self):
        if self.model is None: # Load model only if not already loaded
            self.model = keras.models.load_model(config.MODEL_PATH)
        if self.class_names is None: # Load class names only if not already loaded
            with open(config.CLASS_NAMES_PATH, 'r') as f:
                self.class_names = json.load(f)

    def preprocess_image_for_prediction(self, image_input, target_size=(224, 224)):
        if isinstance(image_input, str):
            # Modern Keras 3 native loader
            img = keras.utils.load_img(image_input, target_size=target_size)
        else:
            image_input.seek(0) 
            img_bytes = image_input.read()
            img = keras.utils.load_img(io.BytesIO(img_bytes), target_size=target_size)

        # Convert the image to a NumPy array natively
        img_array = keras.utils.img_to_array(img)

        # Keep your original [0, 1] scaling logic for the custom Rice model
        img_array = img_array / 255.0

        # Add batch dimension
        processed_img = np.expand_dims(img_array, axis=0)

        return processed_img


        
    def predict(self, image_input):
        """
        Makes a prediction on a single image using the loaded model.

        Args:
            image_input (str or file-like object): The path to the input image file or a file-like object.

        Returns:
            tuple: A tuple containing:
                   - str: The predicted class name.
                   - numpy.ndarray: The prediction probabilities for all classes.
        """
        self.load_model() # Ensure model and class names are loaded

        # Preprocess the image using the updated function
        processed_image = self.preprocess_image_for_prediction(image_input)

        # Make prediction
        predictions = self.model.predict(processed_image)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = self.class_names[predicted_class_index]

        return predicted_class_name, predictions


class Vehicle_Type_Clf():
    """
    A class for classifying vehicle types (e.g., bike or car) using a pre-trained Keras model.
    Loads the model and class names from paths specified in the config module.
    """

    def __init__(self):
        self.model = None
        self.class_names = None

    def load_model(self):
        """
        Loads the Keras model and class names from the specified paths in config.
        Models and class names are loaded only once.
        """
        if self.model is None:
            print(f"DEBUG: Attempting to load vehicle model from: {config.MODEL2_PATH}")
            self.model = keras.models.load_model(config.MODEL2_PATH)
        if self.class_names is None:
            with open(config.CLASS_NAMES2_PATH, 'r') as f:
                self.class_names = json.load(f)

    def preprocess_image_for_prediction(self, image_input, target_size=(224, 224)):
        if isinstance(image_input, str):
            img = image_utils.load_img(image_input, target_size=target_size)
        else:
            image_input.seek(0)
            img_bytes = image_input.read()
            img = image_utils.load_img(io.BytesIO(img_bytes), target_size=target_size)

        img_array = image_utils.img_to_array(img)
        
        # --- FIX HERE: Use standard VGG16 preprocessing instead of division ---
        # First expand dimensions to match the expected batch size structure (1, 224, 224, 3)
        processed_img = np.expand_dims(img_array, axis=0)
        
        # Apply the exact scaling/channel shifting used during the training phase
        processed_img = keras.applications.vgg16.preprocess_input(processed_img)

        return processed_img

    def predict(self, image_input):
        """
        Makes a prediction on a single image using the binary VGG16 model configuration.
        """
        self.load_model()

        processed_image = self.preprocess_image_for_prediction(image_input)

        # predictions shape is (1, 1) due to single neuron sigmoid output
        raw_prediction = self.model.predict(processed_image)[0][0]

        # Binary Threshold evaluation mapping
        if raw_prediction > 0.5:
            predicted_class_name = "Car"
            # Format as an array matching what the frontend visualization expects
            probabilities = np.array([[1.0 - raw_prediction, raw_prediction]])
        else:
            predicted_class_name = "Bike"
            probabilities = np.array([[1.0 - raw_prediction, raw_prediction]])

        return predicted_class_name, probabilities
