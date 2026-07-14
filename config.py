import os

FLASK_HOST = '0.0.0.0'
FLASK_PORT = 8000

MONGO_PASSWORD = 'test123$'
MONGO_URL = f'mongodb+srv://rishabhp:{MONGO_PASSWORD}@docdb-cluster-20260630-0923.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000'
db_name = 'image_clf_db'
collection_data1 = 'rice_img_clf'
collection_data2 = 'vehicle_img_clf'
collection_user = 'clf_users'


MODEL_PATH = os.path.join(os.getcwd(), 'artifacts', 'rice_classifier_vgg16.keras')
CLASS_NAMES_PATH = os.path.join(os.getcwd(), 'artifacts', 'class_names.json')

MODEL2_PATH = os.path.join(os.getcwd(), 'artifacts', 'car_bike_classifier_vgg16.keras')
CLASS_NAMES2_PATH = os.path.join(os.getcwd(), 'artifacts', 'car_bike_class_names.json')

