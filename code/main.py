import audio
import soundDetect
import time
if __name__ == "__main__":
    mic = audio.Mic()
    process = soundDetect.SoundProcess()
    detect = soundDetect.SoundDetect()
    
    flag = False
    barkcount = 0
    while 1:
        mic.Record(4)
        feature = process.Extract_feature('record.wav')
        result = detect.Prediction(feature)
        
        if result == True:
            if flag == True:
                barkcount += 1
            flag = True
        else:
            if flag == True:
                barkcount = 0
            flag = False
        
        if barkcount == 15:
            print("강아지가 짖는중!")
            barkcount = 0
        