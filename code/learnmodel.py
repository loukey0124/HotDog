import os
import librosa
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

MAX_PAD_LEN = 174

def ExtractFeature(fileName):
  print('filename :', fileName)
  
  try:
    audio, sampleRate = librosa.load(fileName, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sampleRate, n_mfcc=70)
    padWidth = MAX_PAD_LEN - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0,0), (0, padWidth)), mode='constant')
    
  except Exception as e:
    print("file parsing Error", fileName)
    print(e)
    return None
  
  return mfccs


def LoadData(read):
  if read == True:
    dataSetPath = '../UrbanSound8K/audio/'
    metaData = pd.read_csv('../Urbansound8K/metadata/UrbanSound8K.csv')
    features = []
  
    for index, row in metaData.iterrows():
      fileName = os.path.join(os.path.abspath(dataSetPath),
                              'fold'+str(row["fold"])+'/', str(row["slice_file_name"]))
    
      classLabel = row["classID"]
      data = ExtractFeature(fileName)
    
      features.append([data, classLabel])
    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])
    
    featuresdf.to_pickle("featuresdf.pkl")
  else:
    featuresdf = pd.read_pickle("featuresdf.pkl")
  
  return featuresdf

def SetModel(featuresdf, batch_size, epochs):
  physical_devices = tf.config.list_physical_devices('GPU')
  try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
  except:
    pass
  
  X = np.array(featuresdf.feature.tolist())
  Y = np.array(featuresdf.class_label.tolist())
  
  le = LabelEncoder()
  yy = to_categorical(le.fit_transform(Y))
  
  x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)
 
  n_columns = 174
  n_row = 70
  n_channels = 1
  n_classes = 10
  
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
  
  learningRate = 0.001
  opt = keras.optimizers.Adam(learning_rate=learningRate)
  
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
  mc = ModelCheckpoint('best_model.h5', monitor='var_loss', mode='min', save_best_only=True)
  
  history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
                      shuffle=True, validation_data=(x_test, y_test), callbacks=[es, mc])

  print('\n# Evaluate on test data')
  results = model.evaluate(x_test, y_test, batch_size=128)
  print('test loss, test acc:', results)
  
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tfliteModel = converter.convert()
  open("SoundModel.tflite", "wb").write(tfliteModel)
  
  return history

def PlotHistory(history):
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
  
if __name__ == "__main__":
  print("load wav?")
  sel = input()
  
  if sel == 'n':
    featuresdf = LoadData(False)
  else:
    featuresdf = LoadData(True)
    
  history = SetModel(featuresdf, 128, 1)
  PlotHistory(history)