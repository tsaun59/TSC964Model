from keras.preprocessing.image import ImageDataGenerator



#preprocessing | for training, we can augment the image data to essentially "fake" a larger dataset
def PreprocessData(batchSize=32, iHeight=160, iWidth=160, dir='data/', firstTraining = True):
    # Data augmentation for training set

    train_datagen = None

    #First training image augmentation. (Evaluator: feel free to experiment)
    if firstTraining:
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=50,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            shear_range=0.2,
            brightness_range=[0.6, 1.4],
            horizontal_flip=True,
            fill_mode='nearest'
        )      
    else:
        #Further training. (Evaluator: feel free to experiment)
        train_datagen = ImageDataGenerator(
            # rescale=1./255,
            # validation_split=0.2,
            # rotation_range=90, 
            # width_shift_range=0.3, 
            # height_shift_range=0.3, 
            # zoom_range=0.3,
            # shear_range=0.3, 
            # brightness_range=[0.5, 1.5], 
            # horizontal_flip=True,
            # fill_mode='nearest'
                   
            rescale=1./255,
            validation_split=0.2,
            rotation_range=5, 
            width_shift_range=0.02, 
            height_shift_range=0.02, 
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    # No augmentation needed for validation set
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    trainingData = train_datagen.flow_from_directory(
        dir,
        target_size=(iHeight, iWidth),
        batch_size=batchSize,
        class_mode='categorical',
        subset='training'
    )

    valData = val_datagen.flow_from_directory(
        dir,
        target_size=(iHeight, iWidth),
        batch_size=batchSize,
        class_mode='categorical',
        subset='validation'
    )

    return trainingData, valData