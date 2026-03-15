#!/usr/bin/env python
# coding: utf-8

# In[4]:

import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import sqlite3
from datetime import datetime
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons'
          }
conn = sqlite3.connect("predictions1.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions1 (
id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT,
    predicted_class TEXT,
    confidence REAL,
    timestamp TEXT
)
""")


conn.commit()
model = load_model("traffic_signals.keras")

st.title("Traffic Sign Recognition")

os.makedirs("images", exist_ok=True)


img_file_buffer = st.camera_input("Take a picture")
filename=f"images/img_{datetime.now().timestamp()}.jpg"

if img_file_buffer is not None:

    img = Image.open(img_file_buffer)

    st.image(img, caption="Captured Image")
    
    img.save(filename)
    
    img = img.resize((30,30))
    img=np.array(img)
    img=img/255
    img = np.expand_dims(img, axis=0)

    predicted = model.predict(img)
    label = np.argmax(predicted)
    confidence=float(np.max(predicted))
    cursor.execute("""INSERT INTO predictions1 (image_path,predicted_class, confidence, timestamp)VALUES (?, ?, ?,?)""",
    (
        filename,
        classes[label],
        confidence,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    st.success(f"Prediction: {classes[label]}")
    


# In[ ]:




