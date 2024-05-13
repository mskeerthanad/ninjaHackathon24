from flask import Flask,request, jsonify
from flask_cors import CORS
from markupsafe import Markup
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from croprecommender import model_prediction
import numpy as np
import base64
from io import BytesIO
from quantityPrice import get_modal_price, get_average_total_arrival
from datetime import datetime
app = Flask(__name__)
CORS(app, origins="*")

@app.post("/requirement")
def prediction():
    return "Predicted Successfully"
# Importing essential libraries and modules


app = Flask(__name__)
CORS(app)


disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

def predict_image(img, model=disease_model):
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    img = Image.open(io.BytesIO(img))
    img_t = transform(img)
    img_u = torch.unsqueeze(img_t, 0)
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]
        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None




@app.route("/crop-predict", methods=["POST"])
def crop_prediction():
    print("In crop-predict")
    print("Inside POST")
    print(request.json)  # Print the JSON payload received in the request body
    data = request.json
    N = int(data['nitrogen'])
    P = int(data['phosphorous'])
    K = int(data['potassium'])
    ph = float(data['ph'])
    rainfall = float(data['rainfall'])
    city = data['city']
    
    if weather_fetch(city) is not None:
        temperature, humidity = weather_fetch(city)
        data_array = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = model_prediction(data_array)
        final_prediction = my_prediction.tolist()  # Convert numpy array to list
        print("Final Prediction:", final_prediction)
        return jsonify({"prediction": final_prediction})
    else:
        return jsonify({"error": "Weather data not available for the city"})


@app.route('/disease-predict', methods=['GET','POST'])
def disease_prediction():
    print("In Disease Predictor")
    print(request.files, "Printing Request Files")
    if 'file' not in request.files:
        print("Inside If not file")
        return redirect(request.url)

    file = request.files.get('file')
    print(file)
    if not file:
        print("Inside not file")
        return None
    try:
        print("Prediction Try:")
        img_binary = file.read()
        img_bytes = BytesIO(img_binary)
        
        # Convert base64 encoded string into bytes
        base64_image = img_bytes.getvalue().decode('utf-8').split(",")[1]
        img_bytes = base64.b64decode(base64_image.encode('utf-8'))
        print(img_bytes)
        prediction = predict_image(img_bytes)
        prediction = Markup(str(disease_dic[prediction]))
        return prediction
    except Exception as e:
        print("Exception:", e)
        pass

    return None

@app.route('/price', methods=['POST'])
def get_price():
    data = request.json
    print("crop-demand", data)
    inputDate = data['date_str']
    state= data['state']
    district= data['district']
    commodity= data['commodity']
    print("Hello World",inputDate,state,district,commodity)
    try: 
        price = get_modal_price(inputDate, state, district, commodity)
        print("Price", price)
        return jsonify({"price": price})  
    except Exception as e:
        print("Exception:", e)
        return jsonify({"error": str(e)})  
         
@app.route("/crop-demand", methods=["POST"])
def get_average_arrival():
    data = request.json
    inputDate = data['date_str']
    state= data['state']
    district= data['district']
    res= get_average_total_arrival(inputDate, state, district)
    print("Res ponse ", res)
    return jsonify({"demand": res})


@app.route('/home/')  
def home():
    return "Home Page"

if __name__ == "__main__":
    app.run(port=8000)
