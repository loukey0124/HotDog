import tflite_runtime.interpreter as tflite
import numpy as np
import librosa
import os

import audio

class SoundProcess:
    def __init__(self):
        self.MAX_PAD_LEN = 174
        
    def Extract_feature(self, fileName):
        try:
            audio, sampleRate = librosa.load(path="record.wav", res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio, sr=sampleRate, n_mfcc=50)
            padWidth = self.MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(array=mfccs, pad_width=((0,0), (0, padWidth)), mode='constant')
            
        except Exception as e:
            print("File parsing Error :", fileName)
            print(e)
            return None
        
        os.remove(fileName)
        return mfccs

class SoundDetect:
    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path='../model/SoundModel159-256-50.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def Prediction(self, feature):
        data = np.reshape(feature, [-1, 50, 174, 1])
        self.interpreter.set_tensor(self.input_details[0]['index'], data)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])
        print(out)
        if out.argmax(axis=1) == 3 and out[0, 3] > 0.9:
            print("!!!!!!!!!")
            return True
        else:
            return False
    
if __name__ == '__main__':
    n = int(input())