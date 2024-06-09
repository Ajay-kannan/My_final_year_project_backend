from flask import Flask, request, jsonify 
from flask_cors import CORS
import predict
import chatbot

app = Flask(__name__)
CORS(app) 

@app.route("/predict", methods=["POST"])
def predict_image_class():
    if request.method == "POST":
        print(request.files)
        if "image" not in request.files :
            return jsonify({"error": "Missing image in request."}), 400

        image = request.files["image"]
        try:
            predicted_class_name = predict.predict_class(image)
            
            print("\n ------------------------------- \n")
            print(predicted_class_name)
            print("\n ------------------------------- \n")
            return jsonify({"predicted_class": predicted_class_name})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid request method. Use POST."}), 405


@app.route("/")
def homepage():
    
    return jsonify("hello")


@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        
        data = request.get_json()
        user_question = data.get('user_question')
        
        print("\n ------------------------------- \n")
        print(f"question : {user_question}" )
        print("\n ------------------------------- \n")
        
        answer = chatbot.generate_response(user_question)
        print(answer)
        # Return the answer as JSON response
        return jsonify(answer), 200
    except Exception as e:
        # Log the exception for debugging
        print(f"An error occurred: {e}")
        
        # Return error response if an exception occurs
        return jsonify({'error': 'Internal Server Error'}), 500
    

@app.route('/disease-plant', methods=['POST'])
def process_string():
    try:
        # Get the input string from the request
        input_string = request.json.get('input_string')

        # Process the input string (for example, you can manipulate it or perform some operation)

        prompt = "plant is affected by "+ input_string +" find remedies."
        answer = chatbot.generate_response(prompt)
        
        print(answer["messageContent"])

        # Create a JSON response with the processed string
        response = {'processed_string': answer["messageContent"]}

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Enable debug mode to see detailed error messages



# pattern = r'__(.*)$'
#             match = re.search(pattern, predicted_class_name)
#             if match:
#                 predicted_class_name = match.group(1)
#             predicted_class_name = string_data + '__' + predicted_class_name





