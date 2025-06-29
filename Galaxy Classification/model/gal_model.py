import tensorflow as tf
import numpy as np 

import random

def gal_predictor():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(96, 96, 3)),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    # import matplotlib.pyplot as plt
    # plt.imshow(img[0])
    # plt.show()

    return model
    # model.fit(Train_data, epochs=10)
    # pred = model.predict(img)
    # predicted_class = np.argmax(pred) 
    # class_names = ['Elliptical', 'Spiral', 'Barred Spiral']
    # print("Predicted Galaxy Type:", class_names[predicted_class])


    # loss, accuracy = model.evaluate(Test_data)
    # print(f"Test Accuracy: {accuracy:.4f}")