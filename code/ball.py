from concurrent.futures import thread
from motordriver import MotorDriver
from servo import Servo

import spidev
from time import sleep
from threading import Thread, Event

class Ball:
    event = Event()
    
    def __init__(self, channel):
        self.motor = MotorDriver(21, 7, 20, 16, 12, 1)
        self.motor.Stop()
        
        self.servo = Servo(15)
        self.servo.Activate(5)
        sleep(1)
        self.servo.Activate(0)
        
        self.channel = channel
        
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 1350000
    
    def AnalogRead(self):
        r = self.spi.xfer2([1, (8 + self.channel) << 4, 0])
        adc_out = ((r[1]&3) << 8) + r[2]
        
        return adc_out
    
    def PlayBall(self):
        isActivate = False
        
        while True:
            if self.event.is_set():
                self.motor.Stop()
                self.servo.Activate(5)
                sleep(1)
                self.servo.Activate(0)
                return
            
            read = self.AnalogRead()
            print(read)
            if read < 70: # 공이 있을때 (어두움)
                print("open")
                self.motor.Right()
                sleep(1)
                self.servo.Activate(12)
                isActivate = True
            else: # 공이 없을때(밝음)
                print("close")
                if isActivate == True:
                    self.motor.Stop()
                    self.servo.Activate(5)
                    isActivate = False
                else:
                    self.motor.Stop()
                    self.servo.Activate(0)
                
            sleep(1)
    
    def Activate(self):
        self.t = Thread(target=self.PlayBall, daemon=True)
        self.event.clear()
        self.t.start()
    
    def Deactivate(self):
        self.event.set()
        sleep(1)
        
if __name__ == "__main__":
    ball = Ball(1)
    ball.Activate()
    sleep(30)
    ball.Deactivate()
    sleep(2)        