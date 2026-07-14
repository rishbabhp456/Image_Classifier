# Rice Image Classification (Rice_Img_CLF)

This project implements a machine learning model for classifying rice images, deployed as a web API using Flask. It leverages TensorFlow for model building, PyMongo for database interactions, and is designed for deployment on Azure Cloud.

## Table of Contents
- [Features](#features)
- [Project Flow](#project-flow)
- [Local Setup](#local-setup)
- [Project Structure](#project-structure)
- [Azure Deployment Guide](#azure-deployment-guide)
  - [Prerequisites](#prerequisites)
  - [1. Set up Azure Cosmos DB (MongoDB API)](#1-set-up-azure-cosmos-db-mongodb-api)
  - [2. Set up Azure Blob Storage for Models](#2-set-up-azure-blob-storage-for-models)
  - [3. Deploy to Azure App Service](#3-deploy-to-azure-app-service)
- [Usage Examples (API)](#usage-examples-api)
- [Requirements](#requirements)

## Features
- **Image Classification**: Utilizes TensorFlow for building and training a rice image classification model.
- **Web API**: Provides a RESTful API using Flask to serve predictions from the trained model.
- **Database Integration**: Interacts with MongoDB (via PyMongo) for potential data storage, logging, or metadata management.
- **Production-Ready**: Configured with Gunicorn for robust production deployment.
- **Cloud Deployment**: Comprehensive guide for deploying the application to Azure Cloud services.

## Project Flow
1.  **Model Training**: A TensorFlow model is trained on a dataset of rice images to learn classification patterns. The trained model is saved (e.g., in HDF5 format).
2.  **Model Loading**: The Flask web application loads the pre-trained model upon startup.
3.  **Inference API**: The application exposes an API endpoint (`/predict`) that accepts image data (e.g., via a POST request).
4.  **Prediction**: Upon receiving an image, the Flask app preprocesses it, feeds it to the loaded TensorFlow model, and returns the classification prediction.
5.  **Data Management**: MongoDB can be used to store inference requests, results, user data, or other application-specific information.

## Local Setup
To run this project locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <https://github.com/your-username/Rice_Img_CLF.git>
    cd Rice_Img_CLF
    ```

2.  **Create a virtual environment and activate it**:
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare your trained model**:
    Place your trained TensorFlow model (e.g., `rice_classifier.h5`) in the `model/` directory. If you don't have one, you'll need to train it first or obtain a pre-trained model.

5.  **Set up MongoDB**:
    Ensure you have a MongoDB instance running (locally or remotely). Update the MongoDB connection string in your application's configuration (e.g., `app.py` or an environment variable).

6.  **Run the Flask application locally**:
    ```bash
    # Set environment variables (example, adjust as needed)
    export FLASK_APP=app.py
    export MONGODB_URI="mongodb://localhost:27017/rice_db"
    export MODEL_PATH="model/rice_classifier.h5"
    
    flask run
    ```
    Alternatively, for a production-like local environment:
    ```bash
    gunicorn -w 4 -b 0.0.0.0:8000 app:app
    ```
    The application should now be accessible at `http://localhost:5000` (Flask default) or `http://localhost:8000` (Gunicorn).

## Project Structure
```
Rice_Img_CLF/
├── .venv/                  # Python virtual environment
├── app.py                  # Main Flask application file (API endpoints, model loading)
├── model/                  # Directory to store trained ML models
│   └── rice_classifier.h5  # Example: A trained Keras/TensorFlow model
├── requirements.txt        # Python dependencies
├── README.md               # Project README file
└── ...                     # Other potential files (e.g., data processing scripts, utility functions)
```

## Azure Deployment Guide
This guide outlines the steps to deploy the Rice Image Classification application to Azure using Azure App Service, Azure Cosmos DB (MongoDB API), and Azure Blob Storage.

### Prerequisites
-   An active Azure subscription.
-   Azure CLI installed and configured.
-   Git installed.
-   Your trained model file (e.g., `rice_classifier.h5`).

### 1. Set up Azure Cosmos DB (MongoDB API)
This will host your MongoDB database.

1.  **Create a Cosmos DB account**:
    ```bash
    az cosmosdb create --name <your-cosmosdb-account-name> --resource-group <your-resource-group> --kind MongoDB --locations "East US"=0
    ```
    Replace `<your-cosmosdb-account-name>` and `<your-resource-group>`.

2.  **Retrieve the connection string**:
    ```bash
    az cosmosdb keys list --name <your-cosmosdb-account-name> --resource-group <your-resource-group> --type connection-strings --query connectionStrings[0].connectionString --output tsv
    ```
    Save this connection string; you'll need it for your App Service environment variables. It will look something like `mongodb://<account-name>:<password>@<account-name>.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@<account-name>@`.

### 2. Set up Azure Blob Storage for Models
This will store your trained machine learning model.

1.  **Create a Storage Account**:
    ```bash
    az storage account create --name <your-storage-account-name> --resource-group <your-resource-group> --location "East US" --sku Standard_LRS
    ```

2.  **Create a Blob Container**:
    ```bash
    az storage container create --name models --account-name <your-storage-account-name> --public-access off
    ```

3.  **Upload your trained model**:
    ```bash
    az storage blob upload --container-name models --file model/rice_classifier.h5 --name rice_classifier.h5 --account-name <your-storage-account-name>
    ```

4.  **Get the Storage Account connection string**:
    ```bash
    az storage account show-connection-string --name <your-storage-account-name> --resource-group <your-resource-group> --query connectionString --output tsv
    ```
    Save this connection string. Your Flask app will use it to download the model.

### 3. Deploy to Azure App Service
This will host your Flask web application.

1.  **Create an Azure Web App**:
    ```bash
    az webapp create --resource-group <your-resource-group> --plan <your-app-service-plan> --name <your-webapp-name> --runtime "PYTHON|3.9"
    ```
    Choose an appropriate Python version (e.g., `PYTHON|3.9`).

2.  **Configure environment variables**:
    Set the MongoDB connection string and Blob Storage connection string as application settings. Your `app.py` should be configured to read these.
    ```bash
    az webapp config appsettings set --resource-group <your-resource-group> --name <your-webapp-name> --settings MONGODB_URI="<your-cosmosdb-connection-string>" AZURE_STORAGE_CONNECTION_STRING="<your-storage-connection-string>" MODEL_BLOB_NAME="rice_classifier.h5" MODEL_CONTAINER_NAME="models"
    ```

3.  **Configure Gunicorn startup command**:
    Azure App Service needs to know how to start your application.
    ```bash
    az webapp config set --resource-group <your-resource-group> --name <your-webapp-name> --startup-file "gunicorn --bind 0.0.0.0 --timeout 600 app:app"
    ```
    Adjust `--timeout` as needed for model loading or long-running requests.

4.  **Deploy your code**:
    You can deploy using Git, Azure DevOps, or local Git deployment. For simplicity, here's local Git deployment:
    ```bash
    az webapp deployment user set --username <your-git-username> --password <your-git-password>
    az webapp deployment source config-local-git --name <your-webapp-name> --resource-group <your-resource-group> --query scmUri --output tsv
    # Copy the scmUri, then add it as a remote to your local git repo
    git remote add azure <scmUri>
    git push azure master
    ```
    Azure will automatically install dependencies from `requirements.txt` and start your app.

## Usage Examples (API)
Once deployed, you can interact with your API. Assuming your Flask app exposes a `/predict` endpoint for image classification:

```python
import requests
import json

app_url = "https://<your-webapp-name>.azurewebsites.net" # Replace with your Azure App Service URL

# Example: Sending an image for prediction
# You would typically read an image file and send its bytes or base64 encoded string
dummy_image_data = "base64_encoded_image_string_here" # Replace with actual image data

response = requests.post(f"{app_url}/predict", json={"image": dummy_image_data})

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)
```

## Requirements
The project dependencies are listed in `requirements.txt`:
```
# --- Core Machine Learning & Data Inference ---
tensorflow==2.16.1
numpy>=1.23.5,<2.0.0
pandas==2.2.2
scikit-learn==1.4.2
h5py==3.11.0
Pillow==10.3.0

# --- Web Framework & API Ecosystem ---
Flask==3.0.3
Werkzeug==3.0.3

# --- Database Drivers ---
pymongo==4.7.2

# --- Production Deployment Gateway ---
gunicorn==22.0.0
```

<!--
[PROMPT_SUGGESTION]Provide an example `app.py` file that implements the Flask API and model loading described in the README.[/PROMPT_SUGGESTION]
[PROMPT_SUGGESTION]How can I set up CI/CD for this project to automatically deploy to Azure App Service on Git push?[/PROMPT_SUGGESTION]