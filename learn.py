import os
import librosa
import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

max_pad_len = 174

def extract_feature(file_name):
    print('file name :', file_name)
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        print(e)
        return None
    
#     return padded_mfccs
    return mfccs

# # Set the path to the full UrbanSound dataset 
# fulldatasetpath = '/content/drive/MyDrive/UrbanSound8K/UrbanSound8K/audio/'
# metadata = pd.read_csv('/content/drive/MyDrive/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv')
# features = []

# # Iterate through each sound file and extract the features 
# for index, row in metadata.iterrows():
#     file_name = os.path.join(os.path.abspath(fulldatasetpath),
# 				'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    
#     class_label = row["classID"]
#     data = extract_feature(file_name)
    
#     features.append([data, class_label])

# # Convert into a Panda dataframe 
# featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

# # 피클로 데이터 저장
# featuresdf.to_pickle("featuresdf.pkl")

# 피클 데이터 로드
featuresdf = pd.read_pickle("featuresdf.pkl")

from tensorflow.keras.utils import to_categorical

X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

n_columns = 174
n_row = 40       
n_channels = 1
n_classes = 10

# input shape 조정
# cpu를 사용해서 수행한다
with tf.device('/cpu:0'):
    x_train = tf.reshape(x_train, [-1, n_row, n_columns, n_channels])
    x_test = tf.reshape(x_test, [-1, n_row, n_columns, n_channels])

model = keras.Sequential()

model.add(layers.Conv2D(input_shape=(n_row, n_columns, n_channels), filters=16, kernel_size=2, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(kernel_size=2, filters=32, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(kernel_size=2, filters=64, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(kernel_size=2, filters=128, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.2))

model.add(layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.summary()

training_epochs = 72
num_batch_size = 128

learning_rate = 0.001
opt = keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# from keras.callbacks import EarlyStopping

# es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0,patience=5)

history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=training_epochs, shuffle=True, validation_data=(x_test, y_test))

import matplotlib.pyplot as plt
    
def plot_history(history):
  fig, loss_ax = plt.subplots()
  acc_ax = loss_ax.twinx()

  loss_ax.plot(history.history['loss'], 'y', label = 'train loss')
  loss_ax.plot(history.history['val_loss'], 'r', label = 'valid loss')

  acc_ax.plot(history.history['accuracy'], 'b', label = 'train accuracy')
  acc_ax.plot(history.history['val_accuracy'], 'g', label = 'valid accuracy')

  loss_ax.set_xlabel('epoch')
  loss_ax.set_ylabel('loss')
  acc_ax.set_ylabel('accuracy')

  loss_ax.legend(loc='upper left')
  acc_ax.legend(loc='lower left')

  plt.show()

plot_history(history)

model.save("sound_classifier_model")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("SoundModel.tflite", "wb").write(tflite_model)