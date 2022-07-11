from flask import Flask,request,jsonify
import numpy as np
from flask_mysqldb import MySQL
from tensorflow import keras

import firebase_admin
from firebase_admin import credentials,storage
import numpy as np
import cv2
cred = credentials.Certificate("andriodminiproject-firebase-adminsdk-wgda3-a7af8ed974.json")
app=firebase_admin.initialize_app(cred,{ "storageBucket":"andriodminiproject.appspot.com"})

bucket=storage.bucket()

import os
model = keras.models.load_model('Mymodel.h5')



li = ['Apple Apple scab',
 'Apple Black_rot',
 'Apple Cedar_apple_rust',
 'Apple healthy',
 'Blueberry healthy',
 'Cherry (including_sour) Powdery_mildew',
 'Cherry (including_sour) healthy',
 'Corn (maize) Cercospora_leaf_spot Gray_leaf_spot',
 'Corn (maize) Common_rust_',
 'Corn (maize) Northern_Leaf_Blight',
 'Corn (maize) healthy',
 'Grape Black rot',
 'Grape Esca (Black_Measles)',
 'Grape Leaf blight (Isariopsis_Leaf_Spot)',
 'Grape healthy',
 'Orange Haunglongbing (Citrus_greening)',
 'Peach Bacterial spot',
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

app = Flask(__name__)


app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'earthquake'

mysql = MySQL(app)



@app.route('/')
def index():
    return "Hi server connected suucesfully"


   


@app.route("/predict", methods = ['POST'])
def predict():
    
    if request.method == 'POST':
        file = request.form['file']
        blob=bucket.get_blob(file)
        arr=np.frombuffer(blob.download_as_string(),np.uint8)
        img_path = cv2.imdecode(arr, cv2.COLOR_BGR2BGR555) 
        #filename = file.filename
        #file_path = os.path.join("static", filename)                       
        #ile.save(file_path)
        new_img = cv2.resize(img_path,(224,224))
        img = keras.utils.img_to_array(new_img)
        img = np.expand_dims(img, axis=0)
        img = img/255   
        prediction = model.predict(img)
        d = prediction.flatten()
        j = d.max()
       
        for index,item in enumerate(d):
            if item == j:
                class_name = li[index]
    return str(class_name)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
