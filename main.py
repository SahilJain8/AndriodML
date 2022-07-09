
from flask import Flask,request,jsonify
import numpy as np
from flask_mysqldb import MySQL
from tensorflow import keras

import os
model = keras.models.load_model('Mymodel.h5')



li = ['Apple___Apple_scab',
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

app = Flask(__name__)


app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'earthquake'

mysql = MySQL(app)



@app.route('/')
def index():
    return "Hello world"


   


@app.route("/predict", methods = ['POST'])
def predict():
    
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join("static", filename)                       
        file.save(file_path)
        new_img = keras.utils.load_img(file_path, target_size=(224, 224))
        img = keras.utils.img_to_array(new_img)
        img = np.expand_dims(img, axis=0)
        img = img/255   
        prediction = model.predict(img)
        d = prediction.flatten()
        j = d.max()
       
        for index,item in enumerate(d):
            if item == j:
                class_name = li[index]
       


















                            

        
       
        
    return jsonify({'The condition is':str(class_name)})


if __name__ == "__main__":
    app.run()
