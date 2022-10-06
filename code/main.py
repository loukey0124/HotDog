import audio
import soundDetect

if __name__ == "__main__":
    mic = audio.Mic()
    process = soundDetect.SoundProcess()
    detect = soundDetect.SoundDetect()
    
    count = 0
    dogcount = 0
    while 1:
        mic.Record(4)
        feature = process.Extract_feature('record.wav')
        result = detect.Prediction(feature)
        
        if result == True:
            dogcount += dogcount
        
        if count == 30:
            count = 0
            dogcount = 0
            
        if dogcount == 15:
            print("강아지가 짖는중!")
        