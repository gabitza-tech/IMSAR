from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        #initialize model 
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channel first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        #we add layers
        model.add(Conv2D(20,(5,5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        #2nd part
        model.add(Conv2D(50,(5,5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        #FC layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        #softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        #return the constructed network architecture
        return model