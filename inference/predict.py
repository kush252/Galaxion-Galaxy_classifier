import numpy as np
import cv2
from utils import pre_proc
from model.gal_model import gal_predictor

Train_data,Test_data=pre_proc()

model=gal_predictor()
model.fit(Train_data, epochs=10)

img_path= input("Kindly provide the file path of the galaxy image for classification:")
if img_path:
    img=cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
    img = cv2.resize(img, (96, 96))
    img = img / 255.0
    img = img.reshape(1, 96, 96, 3).astype("float32")

    pred = model.predict(img)
    predicted_class = np.argmax(pred) 
    class_names = ['Elliptical', 'Spiral', 'Barred Spiral']
    print("Predicted Galaxy Type:", class_names[predicted_class])
else:
    print("Invalid Path")