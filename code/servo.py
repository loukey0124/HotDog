import RPi.GPIO as GPIO
from time import sleep

class Servo:
    def __init__(self, ch):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(ch, GPIO.OUT)
        
        self.servo = GPIO.PWM(ch, 50)
        self.servo.start(0)
        
    
    def Activate(self, duty):
        self.servo.ChangeDutyCycle(duty)
        
    def Clean(self):
        GPIO.cleanup()
        
if __name__ == "__main__":
    servo = Servo(23)
    
    servo.Activate(9)
    sleep(1)
    servo.Activate(3)
    sleep(1)
    servo.Clean()
    del servo