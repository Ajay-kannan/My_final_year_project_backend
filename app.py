from flask import Flask, request, jsonify 
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def load_model_and_preprocess(target_size=(224, 224)):
    """Loads the image classification model and defines a preprocessing function.

    Args:
        model_path (str, optional): Path to the saved model file (default: "model.h5").
        target_size (tuple, optional): Target image size for resizing (default: (224, 224)).

    Returns:
        tensorflow.keras.models.Model: The loaded image classification model.
        function: A function that preprocesses an image for model inference.
    """
    model_path = "C:/Users/Ajay kannan/Desktop/reactprogram/practice/plant_disease_prediction_model.h5"
    model = load_model(model_path)

    def preprocess_image(image_path):
        """Preprocesses an image for model prediction."""
        img = Image.open(image_path)
        img = img.resize(target_size)  # Resize to match model input
        img_array = np.array(img)
        # Normalize pixel values (adjust based on your model's normalization)
        img_array = img_array / 255.0
        img_batch = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_batch

    return model, preprocess_image

model, preprocess_image = load_model_and_preprocess()

@app.route("/predict", methods=["POST"])
def predict_image_class():
    """API endpoint to receive image data and predict class."""

    if request.method == "POST":
        if "image" not in request.files or "string_data" not in request.form:
            return jsonify({"error": "Missing image or string data in request."}), 400

        # Get image and string data from request
        image = request.files["image"]
        string_data = request.form["string_data"]

        try:
            # Preprocess the image
            preprocessed_image = preprocess_image(image.filename)

            # Make prediction using the model
            predictions = model.predict(preprocessed_image)
            predicted_class = np.argmax(predictions[0])  # Get index of most likely class

            # (Optional) Perform additional processing based on string_data
            # if relevant to your prediction logic

            return jsonify({"predicted_class": predicted_class, "string_data": string_data})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid request method. Use POST."}), 405


@app.route("/")
def homepage():
    
    return jsonify("hello world")





# @app.route('/ask', methods=['POST'])
# def ask_question():
#     try:
#         # Get the user_question from the request body
#         data = request.get_json()
#         user_question = data.get('user_question')
        
#         # Perform some processing or logic to get the answer
#         # For demonstration purposes, let's just echo back the question
#         print("\n ------------------------------- \n")
#         print(f"question : {user_question}" )
#         print("\n ------------------------------- \n")
        
#         answer = get_answer(user_question)
        
#         # Return the answer as JSON response
#         return jsonify({'answer': answer}), 200
#     except Exception as e:
#         # Log the exception for debugging
#         print(f"An error occurred: {e}")
        
#         # Return error response if an exception occurs
#         return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Enable debug mode to see detailed error messages






