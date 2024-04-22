import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

class_indices = {"0": "Apple__black_rot", "1": "Apple__healthy", "2": "Apple__rust", "3": "Apple__scab", "4": "Cassava__bacterial_blight", "5": "Cassava__brown_streak_disease", "6": "Cassava__green_mottle", "7": "Cassava__healthy", "8": "Cassava__mosaic_disease", "9": "Cherry__healthy", "10": "Cherry__powdery_mildew", "11": "Chili__healthy", "12": "Chili__leaf curl", "13": "Chili__leaf spot", "14": "Chili__whitefly", "15": "Chili__yellowish", "16": "Coffee__cercospora_leaf_spot", "17": "Coffee__healthy", "18": "Coffee__red_spider_mite", "19": "Coffee__rust", "20": "Corn__common_rust", "21": "Corn__gray_leaf_spot", "22": "Corn__healthy", "23": "Corn__northern_leaf_blight", "24": "Cucumber__diseased", "25": "Cucumber__healthy", "26": "Gauva__diseased", "27": "Gauva__healthy", "28": "Grape__black_measles", "29": "Grape__black_rot", "30": "Grape__healthy", "31": "Grape__leaf_blight_(isariopsis_leaf_spot)", "32": "Jamun__diseased", "33": "Jamun__healthy", "34": "Lemon__diseased", "35": "Lemon__healthy", "36": "Mango__diseased", "37": "Mango__healthy", "38": "Peach__bacterial_spot", "39": "Peach__healthy", "40": "Pepper_bell__bacterial_spot", "41": "Pepper_bell__healthy", "42": "Pomegranate__diseased", "43": "Pomegranate__healthy", "44": "Potato__early_blight", "45": "Potato__healthy", "46": "Potato__late_blight", "47": "Rice__brown_spot", "48": "Rice__healthy", "49": "Rice__hispa", "50": "Rice__leaf_blast", "51": "Rice__neck_blast", "52": "Soybean__bacterial_blight", "53": "Soybean__caterpillar", "54": "Soybean__diabrotica_speciosa", "55": "Soybean__downy_mildew", "56": "Soybean__healthy", "57": "Soybean__mosaic_virus", "58": "Soybean__powdery_mildew", "59": "Soybean__rust", "60": "Soybean__southern_blight", "61": "Strawberry___leaf_scorch", "62": "Strawberry__healthy", "63": "Sugarcane__bacterial_blight", "64": "Sugarcane__healthy", "65": "Sugarcane__red_rot", "66": "Sugarcane__red_stripe", "67": "Sugarcane__rust", "68": "Tea__algal_leaf", "69": "Tea__anthracnose", "70": "Tea__bird_eye_spot", "71": "Tea__brown_blight", "72": "Tea__healthy", "73": "Tea__red_leaf_spot", "74": "Tomato__bacterial_spot", "75": "Tomato__early_blight", "76": "Tomato__healthy", "77": "Tomato__late_blight", "78": "Tomato__leaf_mold", "79": "Tomato__mosaic_virus", "80": "Tomato__septoria_leaf_spot", "81": "Tomato__spider_mites_(two_spotted_spider_mite)", "82": "Tomato__target_spot", "83": "Tomato__yellow_leaf_curl_virus", "84": "Wheat__brown_rust", "85": "Wheat__healthy", "86": "Wheat__septoria", "87": "Wheat__yellow_rust"}

def load_model_and_preprocess(target_size=(224, 224)):
   
    model_path = "C:/Users/Ajay kannan/Desktop/reactprogram/practice/plant_disease_prediction_model.h5"
    model = load_model(model_path)

    def preprocess_image(image_path):
        # Load the image
        img = Image.open(image_path)
        # Resize the image
        img = img.resize(target_size)
        # Convert the image to a numpy array
        img_array = np.array(img)
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        # Scale the image values to [0, 1]
        img_array = img_array.astype('float32') / 255.
        return img_array

    return model, preprocess_image




def  predict_class(image):
    model , preprocess_image= load_model_and_preprocess()
    preprocessed_image = preprocess_image(image)

    # Make prediction using the model
    predictions = model.predict(preprocessed_image)
    
    predicted_class_index = np.argmax(predictions, axis=1)[0] # Get index of most likely class
    
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name
