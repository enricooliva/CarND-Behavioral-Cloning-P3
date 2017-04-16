import csv
import cv2
import numpy as np
import os
import random

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.utils import shuffle

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Activation, Reshape, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
    
current_path = './data/IMG/' 

#from https://medium.com/@acflippo/cloning-driving-behavior-by-augmenting-steering-angles-5faf7ea8a125
def perturb_angle(angle):
    new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
    return new_angle
    
def read_image(filename):
     image = cv2.imread(current_path + filename)
     image = image[50:-20, :, :]
     #image = change_brightness(image)
     image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
     #image = cv2.resize(image, (64, 64), cv2.INTER_AREA)
     return image

def generator(samples, batch_size=32):
    correction = 0.25 # this is a parameter to tune
    num_samples = len(samples)
    images = []
    measurements = []
    while 1:
        shuffle(samples)
        for i in range(len(samples)):
            batch_sample = samples[i]
            measurement = float(batch_sample[3])
            if (abs(measurement)>0.85):
                camera = np.random.choice(['center', 'left', 'right'])
                if camera == 'center':
                    filename = batch_sample[0].split('/')[-1]
                    image=read_image(filename)
                    images.append(image)
                elif camera == 'left':
                    filename = batch_sample[1].split('/')[-1]
                    image=read_image(filename)
                    images.append(image)
                    measurement += correction
                elif camera == 'right':
                    filename = batch_sample[2].split('/')[-1]
                    image=read_image(filename)
                    images.append(image)
                    measurement -= correction
                   
                measurement = perturb_angle(measurement)
                measurements.append(measurement)

                if (len(images)==batch_size):
                    yield shuffle(np.array(images), np.array(measurements))
                    images, measurements = ([],[])

                #flip_image
                #if np.random.randint(2) == 1:
                images.append(cv2.flip(image,1))
                measurement = perturb_angle(measurement*-1.0)
                measurements.append(measurement)
                if (len(images)==batch_size):
                    yield shuffle(np.array(images), np.array(measurements))
                    images, measurements = ([],[])

#NVIDIA model
def build_model():

    model = Sequential()
   
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(90,320,3)))

    model.add(Convolution2D(24, 5, 5, activation="elu", subsample=(2, 2), border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(36, 5, 5, activation="elu", subsample=(2, 2), border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(48, 5, 5, activation="elu", subsample=(2, 2), border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(64, 3, 3, activation="elu", border_mode='valid'))
    model.add(Convolution2D(64, 3, 3, activation="elu", border_mode='valid'))
   
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    return model

def train_model(model, train_generator, validation_generator, train_samples, validation_samples):
   
    checkpoint = ModelCheckpoint('model.{epoch:02d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 mode='auto',
                                 period=1
                                )
    
    #if os.path.isfile('model.h5'):
    #    print("load model ")
    #    model.load_weights('model.h5')
    
    model.summary()
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))
   
    history_object = model.fit_generator(train_generator, 
                samples_per_epoch=19200, 
                validation_data=validation_generator, 
                nb_val_samples=3840, nb_epoch=5, verbose=1, callbacks=[checkpoint])

    model.save('model.h5')
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('./temp.png')
    #plt.show()

    
def main():

    lines = []

    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            lines.append(line)

    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=128)
    validation_generator = generator(validation_samples, batch_size=128)

    model = build_model()
    
    train_model(model,train_generator,validation_generator,train_samples,validation_samples)

if __name__ == '__main__':
    main()
