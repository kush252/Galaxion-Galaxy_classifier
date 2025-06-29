import os
import numpy as np
import tensorflow as tf

def gal_dataset():
    img_size = (96, 96)
    batch_size = 64
    limit_train = 20000
    limit_test = 5000  

    def load_dataset(path, limit, shuffle=True):
       
        full_ds = tf.keras.utils.image_dataset_from_directory(
            directory=path,
            label_mode='categorical',
            image_size=img_size,
            batch_size=batch_size,
            shuffle=shuffle
        )

        
        ds = (
            full_ds
            .unbatch()
            .take(limit)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )
        return ds

   
    train_path = input("Enter Training Folder Path")
    test_path  = input("Enter Testing Folder Path")

    if train_path and test_path:
        S69_traindata=load_dataset(train_path,limit=limit_train, shuffle=True)
        S69_testdata=load_dataset(test_path,limit=limit_test, shuffle=True)

        return S69_traindata,S69_testdata
    else:
        print("Invalid Path")

