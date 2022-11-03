import pyaudio
import wave

class Mic:
    def __init__(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.FILENAME = "record.wav"
        self.p = pyaudio.PyAudio()
    
    def __del__(self):
        self.p.terminate()
    
    def Record(self, RecordSecond):
        print("Record Start.")
        frames = []
        stream = self.p.open(rate=self.RATE, channels=self.CHANNELS, format=self.FORMAT,
                    input=True, frames_per_buffer=self.CHUNK)
        
        for i in range(0, int(self.RATE / self.CHUNK * RecordSecond)):
            data = stream.read(self.CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        print("Record Finished")
        
        wf = wave.open(self.FILENAME, "wb")
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
class Speaker:
    def __init__(self):
        self.CHUNK = 1024
        self.FILENAME = "./data/sound.wav"
        self.p = pyaudio.PyAudio()
        
    def Play(self):
        with wave.open(self.FILENAME, 'rb') as f:
            stream = self.p.open(format = self.p.get_format_from_width(f.getsampwidth()),
                                 channels=f.getnchannels(),
                                 rate = f.getframerate(),
                                 output=True)
            
            data = f.readframes(self.CHUNK)
            
            while data:
                stream.write(data)
                data = f.readframes(self.CHUNK)
            
            stream.stop_stream()
            stream.close()
    
    def __del__(self):
        self.p.terminate()

if __name__ == '__main__':
    n = int(input())
    s = Speaker()
    s.Play()