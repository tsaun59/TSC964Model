# Tamarri Saunders
# 011249329
# WGU Capstone C964
# GameConnect's Final Fantasy Image Recognition Model.


import tensorflow as tf
from keras import layers,models
from imagepreprocessing import PreprocessData
import pickle
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import load_model
import evaluation as eval
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt



#####Settings#####
#
# If interested in BUILDING and TRAINING a NEW model, move any old models in the root to the "oldmodels folder" so they're not overwritten. Then set buildNewModel to True
# If interested in RETRAINING an OLD model, set "modelLoadPath" to the path of the model for retraining.
# If interested in VIEWING an old model structure, set "modelLoadPath" to the path of the model, and the console will display information on the model.
# *To the evaluator* if you decide to build your own model I suggest keeping the model build the same, but changing around the augmentation in imageprocessing.py instead. 
buildNewModel = True
GetModelSummary = False
modelSavePath = "GameConnectGameIdentificationModel.keras"
modelLoadPath = "GameConnectGameIdentificationModel.keras"


earlyStopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
learningRateReduc = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,) 

#
#
#
###################

#building model | basically the building blocks of the model
def BuildingModel(shape=(160,160,3),classCount=3):
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=shape, kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization(name="bn1"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool1"))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', name="conv2", kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization(name="bn2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool2"))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', name="conv3", kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization(name="bn3"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool3"))
    model.add(layers.Dropout(0.2, name="dropout4"))

    # Block 4
    model.add(layers.Conv2D(128, (3, 3), activation='relu', name="conv4",kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization(name="bn4"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool4"))
    model.add(layers.Dropout(0.2, name="dropout3")) #0.2

    # # Block 5 (might get rid of this)
    # model.add(layers.Conv2D(128, (3, 3), activation='relu', name="conv5",kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    # model.add(layers.BatchNormalization(name="bn5"))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool5"))
    # model.add(layers.Dropout(0.3, name="dropout2"))


    # Classifier
    model.add(layers.GlobalAveragePooling2D(name="gap"))
    model.add(layers.BatchNormalization(name="bn6")) 
    model.add(layers.Dense(512, activation='relu', name="dense1"))
    model.add(layers.Dropout(0.6, name="dropout1")) #0.6 -> 0.41, 0.8808 epoch 61
    model.add(layers.Dense(classCount, activation='softmax', name="output"))


    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)

    model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

    return model

#training model | based off the dataset and built model.
def TrainingModel(model, trainingData, valData, epochs=100):
    hist = model.fit(trainingData,epochs=epochs,validation_data=valData,callbacks = [earlyStopping,learningRateReduc])

    with open('training_history.pkl', 'wb') as f:
        pickle.dump(hist.history, f)

    return hist


#main | preprocessing is done here before working with the model
if __name__ == "__main__":



    if GetModelSummary: 
        model = load_model(modelLoadPath)
        model.summary()
        optimizer_config = model.optimizer.get_config()
        print(optimizer_config)
        print(model.loss)

    else:

        if buildNewModel:
            trainingData,valData = PreprocessData(firstTraining=True)

            model = BuildingModel(shape=(160,160,3),classCount=len(trainingData.class_indices))

            hist = TrainingModel(model,trainingData,valData)

            model.save(modelSavePath)
            print("Model Saved")

        else:

            trainingData,valData = PreprocessData(firstTraining=False)

            model = load_model(modelLoadPath)


            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.00)
            optimizer = Adam(learning_rate=1e-4)
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

            hist = TrainingModel(model,trainingData,valData)

            model.save(modelSavePath)
            print("Model Saved")

    