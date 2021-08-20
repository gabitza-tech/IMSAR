from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.applications.vgg16 import VGG16 
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor

from keras_vggface.vggface import VGGFace

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2
#import ArcFace
#from deepface import DeepFace

train_data_dir = "pyimagesearch/datasets/emotion_compilation/train"
validation_data_dir = "pyimagesearch/datasets/emotion_compilation/valid"
test_data_dir = "pyimagesearch/datasets/emotion_compilation/test"

def step_decay(epoch):
    # initialize the base initial learning rate, drop factor, and
    # epochs to drop every
    initAlpha = 0.01
    factor = 0.5
    dropEvery = 5

    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    # return the learning rate
    return float(alpha)

#augmentation on training data
train_datagen = ImageDataGenerator(rescale = 1.0 /255.0, rotation_range =30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")

# this is the augmentation configuration we will use for validation and testing:
# only rescaling
test_datagen = ImageDataGenerator()


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224), # not running this way
    batch_size=24,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224), # not running this way
    batch_size= 24,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224,224),
    batch_size = 24,
    class_mode='categorical'
)

#number of classes and classes dict
classes = train_generator.class_indices
classNames = [str(x) for x in test_generator.class_indices.keys()]
print("Number of classes:{} and their names:{}".format(len(classes),classNames))

# DEFINE MODEL 
##
##
print("[INFO] loading network...")
#baseModel = DeepFace.build_model(model_name='Facenet')
#baseModel = ArcFace.loadModel()
#baseModel.load_weights('arcface_weights.h5')

#baseModel = VGG16(weights="imagenet", input_tensor=Input(shape=(224, 224, 3)), include_top=False)
baseModel = VGGFace(model='resnet50',input_shape=(224, 224, 3),include_top=False)

#for (i, layer) in enumerate(baseModel.layers):
#    print("[INFO] {}\t{}".format(i, layer.__class__.__name__))

#baseModel.summary()
#print(baseModel.layers[0])
#print("Inputs size:{}".format(baseModel.inputs))
#print("Outputs size:{}".format(baseModel.outputs))

class FCHeadNet:
    @staticmethod 
    def build(baseModel, classes, D=1024):
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.6)(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(classes, activation="softmax")(headModel)

        return headModel


#De ce construiesc modelul folosind model cu inputs baseModel si outputs headModel? De ce nu folosesc direct headModel
#Cand foloses headmodel= Dense(...)(headModel), nu inseamna ca adaug pur si simplu un nou layer?
headModel = FCHeadNet.build(baseModel, len(classes))
model = Model(inputs=baseModel.input, outputs=headModel)

for (i, layer) in enumerate(model.layers):
    print("[INFO] {}\t{}".format(i, layer.__class__.__name__))
#model.summary()

# construct the set of callbacks
#for checkpointing weights folder
fname = os.path.sep.join(['weights_warm',"weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
fname2 = os.path.sep.join(['weights_final',"weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
#for plot and json folders
figPath = os.path.sep.join(['output_warm', "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join(['output_warm', "{}.json".format(os.getpid())])
figPath2 = os.path.sep.join(['output_final', "{}.png".format(os.getpid())])
jsonPath2 = os.path.sep.join(['output_final', "{}.json".format(os.getpid())])

checkpoint = ModelCheckpoint(fname, monitor="val_loss", save_best_only=True, mode = "min", verbose=1)
checkpoint2 = ModelCheckpoint(fname2, monitor="val_loss", save_best_only=True, mode="min", verbose=1)

callbacks = [LearningRateScheduler(step_decay), TrainingMonitor(figPath, jsonPath=jsonPath),checkpoint]
callbacks2 = [TrainingMonitor(figPath2, jsonPath=jsonPath2),checkpoint2]

# WARMING UP THE HEAD OF THE MODEL
# BY FREEZING THE BASE
for layer in baseModel.layers:
    layer.trainable = False

# initialize the optimizer and model
opt = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

print("[INFO] training warming the fully connected network...")
model.fit(train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,
    callbacks=callbacks,
    validation_data= validation_generator,
    verbose = 1)

#predictions after warmup
print("[INFO] evaluating after initialization/warm-up...")
predictions = model.predict(test_generator, batch_size=32)
print(classification_report(test_generator.classes, predictions.argmax(axis=1),target_names=classNames))
    
for layer in baseModel.layers[153:]:
    layer.trainable = True

# initialize the optimizer and model
opt = Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

print("[INFO] training fine tuning...")
model.fit(train_generator,
    steps_per_epoch=len(train_generator),
    epochs=80,
    callbacks=callbacks2,
    validation_data= validation_generator,
    verbose = 1)

#predictions after warmup
print("[INFO] evaluating after fine tuning")
predictions = model.predict(test_generator, batch_size=32)
print(classification_report(test_generator.classes, predictions.argmax(axis=1),target_names=classNames))
    
model.save('weights_model.hdf5')    
