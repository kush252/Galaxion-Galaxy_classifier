import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
# Set path to the 227x227 folder
import tensorflow as tf

def gal_dataset():
    img_size = (96, 96)
    batch_size = 64
    limit_train = 20000
    limit_test = 5000  # optional

    def load_dataset(path, limit, shuffle=True):
        # Load full dataset
        full_ds = tf.keras.utils.image_dataset_from_directory(
            directory=path,
            label_mode='categorical',
            image_size=img_size,
            batch_size=batch_size,
            shuffle=shuffle
        )

        # Limit, optimize
        ds = (
            full_ds
            .unbatch()
            .take(limit)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )
        return ds

    # Paths to your train and test directories
    train_path = input("Enter Training Folder Path")
    test_path  = input("Enter Testing Folder Path")

    if train_path and test_path:
        S69_traindata=load_dataset(train_path,limit=limit_train, shuffle=True)
        S69_testdata=load_dataset(test_path,limit=limit_test, shuffle=True)

        return S69_traindata,S69_testdata
    else:
        print("Invalid Path")


    
        
    
    # S227_traindata,S227_testdata=setsgenerator("D:/Kush/2_mon_vac/proj/Galaxy Classification/image_f/images_E_S_SB_227x227_a_03/images_E_S_SB_227x227_a_03_train","D:/Kush/2_mon_vac/proj/Galaxy Classification/image_f/images_E_S_SB_227x227_a_03/images_E_S_SB_227x227_a_03_test")
    # S299_traindata,S299_testdata=setsgenerator("D:/Kush/2_mon_vac/proj/Galaxy Classification/image_f/images_E_S_SB_299x299_a_03/images_E_S_SB_299x299_a_03_train","D:/Kush/2_mon_vac/proj/Galaxy Classification/image_f/images_E_S_SB_299x299_a_03/images_E_S_SB_299x299_a_03_test")

    # S69_train_data=generator_to_dataset(S69_traindata)
    # S69_test_data=generator_to_dataset(S69_testdata)
    # S227_train_data=generator_to_dataset(S227_traindata)
    # S227_test_data=generator_to_dataset(S227_testdata)
    # S299_train_data=generator_to_dataset(S299_traindata)
    # S299_test_data=generator_to_dataset(S299_testdata)

    # combined_train_ds = S69_train_data.concatenate(S227_train_data).concatenate(S299_train_data)
    # combined_test_ds = S69_test_data.concatenate(S227_test_data).concatenate(S299_test_data) 

    


# for i, (image, label) in enumerate(combined_train_ds.take(5)):
#     print(f"Sample {i+1}")
#     print("Image shape:", image.shape)
#     print("Label (one-hot):", label.numpy())
#     print()
