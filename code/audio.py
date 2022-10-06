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

if __name__ == '__main__':
    n = int(input())