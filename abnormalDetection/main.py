import cv2
import numpy as np
import glob
import os
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
#import imutils
from keras.preprocessing.image import img_to_array, load_img

# ****** The porpose of this file is to make a trained model based on Avenue dataset ****** #

store_image = []


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


train_path = '/Users/malakotb/Users/malakotb/Downloads/Book_Recommendation_System/suh/AvenueDataset/training_videos'
fps = 5
train_videos = os.listdir(train_path)
train_images_path = train_path + '/frames'
create_dir(train_images_path)


def store_inarray(image_path):
    image = load_img(image_path)
    image = img_to_array(image)
    image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_AREA)
    gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    store_image.append(gray)


for video in train_videos:
    os.system('ffmpeg -i {}/{} -r 1/{}  {}/frames/%03d.jpg'.format(train_path, video, fps, train_path))
    images = os.listdir(train_images_path)
    for image in images:
        image_path = train_images_path + '/' + image
        store_inarray(image_path)

store_image = np.array(store_image)
a, b, c = store_image.shape
store_image.resize(b, c, a)
store_image = (store_image - store_image.mean()) / (store_image.std())
store_image = np.clip(store_image, 0, 1)
np.save('training.npy', store_image)

stae_model = Sequential()
stae_model.add(
    Conv3D(filters=128, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='valid', input_shape=(227, 227, 10, 1),
           activation='tanh'))
stae_model.add(Conv3D(filters=64, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid', activation='tanh'))
stae_model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', dropout=0.4, recurrent_dropout=0.3,
                          return_sequences=True))
stae_model.add(
    ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', dropout=0.3, return_sequences=True))
stae_model.add(
    ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, return_sequences=True, padding='same', dropout=0.5))
stae_model.add(
    Conv3DTranspose(filters=128, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid', activation='tanh'))
stae_model.add(
    Conv3DTranspose(filters=1, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='valid', activation='tanh'))
stae_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

training_data = np.load('training.npy')
frames = training_data.shape[2]
frames = frames - frames % 10
training_data = training_data[:, :, :frames]
training_data = training_data.reshape(-1, 227, 227, 10)
training_data = np.expand_dims(training_data, axis=4)
target_data = training_data.copy()
epochs = 5
batch_size = 1

callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]


stae_model.fit(training_data, target_data, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

stae_model.save("saved_model.h5")
