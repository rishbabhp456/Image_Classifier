import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses unneeded compiled optimization warnings
from flask import Flask, request, jsonify, render_template
import config
import pymongo
import datetime
from src.utils import Rice_Type_Clf, Vehicle_Type_Clf


client = pymongo.MongoClient(config.MONGO_URL)
db = client[config.db_name]
rice_collection = db[config.collection_data1]
vehicle_collection = db[config.collection_data2]

app = Flask(__name__)


classifier = Rice_Type_Clf()
vehicle_classifier = Vehicle_Type_Clf()

@app.route('/', methods=['GET'])
def render_dashboard():
    """Renders the centralized dashboard platform."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"status": "failure", "message": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "failure", "message": "No selected file"}), 400

    if file:
        try:
            predicted_class, probabilities = classifier.predict(file.stream)
            prob_list = probabilities.tolist()

            log_payload = {
                "filename": file.filename,
                "predicted_class": predicted_class,
                "probabilities": prob_list,
                "timestamp": datetime.datetime.utcnow()
            }
            rice_collection.insert_one(log_payload)

            return jsonify({
                "status": "success",
                "predicted_class": predicted_class,
                "class_names": classifier.class_names,  
                "probabilities": prob_list
            }), 200
        except Exception as e:
            return jsonify({"status": "failure", "message": f"Error processing image: {str(e)}"}), 500
            
    return jsonify({"status": "failure", "message": "An unexpected error occurred"}), 500


@app.route('/predict_vehicle', methods=['POST'])
def predict_vehicle_image():
    if 'image' not in request.files:
        return jsonify({"status": "failure", "message": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "failure", "message": "No selected file"}), 400

    if file:
        try:
            predicted_class, probabilities = vehicle_classifier.predict(file.stream)
            prob_list = probabilities.tolist()

            log_payload = {
                "filename": file.filename,
                "predicted_class": predicted_class,
                "probabilities": prob_list,
                "timestamp": datetime.datetime.utcnow()
            }
            vehicle_collection.insert_one(log_payload)

            return jsonify({
                "status": "success",
                "predicted_class": predicted_class,
                "class_names": vehicle_classifier.class_names,  
                "probabilities": prob_list
            }), 200
        except Exception as e:
            return jsonify({"status": "failure", "message": f"Error processing image: {str(e)}"}), 500
            
    return jsonify({"status": "failure", "message": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=True)