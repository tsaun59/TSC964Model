import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter
import matplotlib.gridspec as gridspec

##### Basic Settings##### 

# To the evaluator I suggest keeping everything here the same
iHeight = 160
iWidth = 160
batchSize = 32
modelPath = "GameConnectGameIdentificationModel.keras"
dataPath = "data/"

##################

with open('training_history.pkl', 'rb') as f:
    history = pickle.load(f)

#loading validation data (dataset)
def GetValidationData(shuff = False):
    imgDataGen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    valData = imgDataGen.flow_from_directory(
        dataPath,
        target_size=(iHeight, iWidth),
        batch_size=batchSize,
        class_mode='categorical',
        subset='validation',
        shuffle=shuff
    )
    return valData

#evaluating model
def EvaluateModel(model, valData):
    valLoss, valAccuracy = model.evaluate(valData)
    print(f"Validation Loss: {valLoss:.4f}")
    print(f"Validation Accuracy: {valAccuracy:.4f}")


#bar chart for data distribution, basically how many images per category were validated
def CreateDataDistributionPlot(valData):
    classIndices = valData.class_indices
    classNames = list(classIndices.keys())
    labelCount = Counter(valData.classes)

    counts = [labelCount[i] for i in range(len(classNames))]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=classNames, y=counts)
    plt.title("Validation Data Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig("evalresults/data_distribution.png", dpi=300)
    plt.show()


#confusion matrix, basically shows how often the model went wrong, and with what.
def CreateConfusionMatrixPlot(model, valData):
    yPred = model.predict(valData)
    yPredClasses = np.argmax(yPred, axis=1)
    yTrue = valData.classes
    labels = list(valData.class_indices.keys())

    confMatrix = confusion_matrix(yTrue, yPredClasses)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confMatrix, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("evalresults/confusion_matrix.png", dpi=300)
    plt.show()

    print(classification_report(yTrue, yPredClasses, target_names=labels))


#line graph, training history, basically it's progress from start to end in training accuracy and validation accuracy.
def CreateTrainingHistoryPlot(history):
    plt.figure(figsize=(10, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("evalresults/training_history.png", dpi=300)
    plt.show()

#grid of examples of predictions the model made.
def ShowRandomPredictions(model, valData, exampleCount=15):
    xBatch, yBatch = valData.next()
    predictions = model.predict(xBatch)

    labels = list(valData.class_indices.keys())

    plt.figure(figsize=(15, 9))  #width, for more items per row

    for i in range(exampleCount):
        plt.subplot(3, 5, i + 1)
        plt.imshow(xBatch[i])
        labelTrue = labels[np.argmax(yBatch[i])]
        labelPred = labels[np.argmax(predictions[i])]
        plt.title(f"T: {labelTrue}\nP: {labelPred}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("evalresults/prediction_samples.png", dpi=300)
    plt.show()


#heatmaps of the validation data, basically shows what pixels were part of the decision making for the categorization
def CreateHeatmap(imgArray, model, lastConvLayerName, predIndex=None):
    gradModel = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(lastConvLayerName).output, model.output]
    )

    with tf.GradientTape() as tape:
        convOutputs, predictions = gradModel(imgArray)
        if predIndex is None:
            predIndex = tf.argmax(predictions[0])
        classChannel = predictions[:, predIndex]

    grads = tape.gradient(classChannel, convOutputs)
    pooledGrads = tf.reduce_mean(grads, axis=(0, 1, 2))
    convOutputs = convOutputs[0]
    heatMap = convOutputs @ pooledGrads[..., tf.newaxis]
    heatMap = tf.squeeze(heatMap)
    heatMap = tf.maximum(heatMap, 0) / tf.math.reduce_max(heatMap)
    return heatMap.numpy()

#generates heatmaps, grid form.
def ShowHeatmaps(model, valData, layer='conv2d_2'):
    xBatch, _ = valData.next()
    exampleCount = 15  # 15 original + heatmap pairs
    cols = 6  # 3 pairs per row 
    rows = 5  # 5 rows total

    plt.figure(figsize=(18, 10))  # width, for more items per row

    for i in range(exampleCount):
        img = xBatch[i]
        imgArray = img[np.newaxis, ...]

        heatmap = CreateHeatmap(imgArray, model, lastConvLayerName=layer)

        colOffset = (i % 3) * 2  
        row_idx = i // 3

        # Subplot index
        orig_index = row_idx * cols + colOffset + 1
        heatmap_index = orig_index + 1

        # Original image
        plt.subplot(rows, cols, orig_index)
        plt.imshow(img)
        plt.title("Original", fontsize=8)
        plt.axis('off')

        # Heatmap image
        plt.subplot(rows, cols, heatmap_index)
        plt.imshow(img)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title("Grad-CAM", fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("evalresults/prediction_heatmaps.png", dpi=300)
    plt.show()



#main | Performs an evaluation on the model located at modelPath
if __name__ == "__main__":
    model = load_model(modelPath)
    valData = GetValidationData()
    valDataShuff = GetValidationData(True)

    EvaluateModel(model, valData)

    CreateDataDistributionPlot(valData)

    CreateConfusionMatrixPlot(model, valData)

    CreateTrainingHistoryPlot(history)

    ShowRandomPredictions(model, valDataShuff)
    
    ShowHeatmaps(model, valDataShuff, layer='conv3')