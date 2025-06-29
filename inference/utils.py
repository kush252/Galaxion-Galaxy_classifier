from model.data import gal_dataset
import tensorflow as tf
def pre_proc():
    Prep_Train_data,Prep_Test_data=gal_dataset()
    Prep_Train_data=Prep_Train_data.batch(64)
    Prep_Test_data=Prep_Test_data.batch(64)

    def normalize_image(image, label):
        image = image / 255.0
        return image, label

    Prep_Train_data = Prep_Train_data.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    Prep_Test_data = Prep_Test_data.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)

    return Prep_Train_data,Prep_Test_data