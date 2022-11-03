import RPi.GPIO as GPIO

class MotorDriver:
    STOP  = 0
    FORWARD  = 1
    BACKWORD = 2

    CH1 = 0
    CH2 = 1

    OUTPUT = 1
    INPUT = 0

    HIGH = 1
    LOW = 0
    
    def __init__(self, ena, enb, in1, in2, in3, in4):
        self.ENA = ena
        self.ENB = enb 

        self.IN1 = in1 
        self.IN2 = in2 
        self.IN3 = in3 
        self.IN4 = in4
        
        GPIO.setmode(GPIO.BCM)
        
        GPIO.setup(self.ENA, GPIO.OUT)
        GPIO.setup(self.IN1, GPIO.OUT)
        GPIO.setup(self.IN2, GPIO.OUT)
        
        GPIO.setup(self.ENB, GPIO.OUT)
        GPIO.setup(self.IN3, GPIO.OUT)
        GPIO.setup(self.IN4, GPIO.OUT)
        
        self.pwmA = GPIO.PWM(self.ENA, 100)
        self.pwmB = GPIO.PWM(self.ENB, 100)
        
        self.pwmA.start(0)
        self.pwmB.start(0)        
        
    def SetMotorContorl(self, pwm, ina, inb, speed, stat):
        pwm.ChangeDutyCycle(speed)  
        
        if stat == self.FORWARD:
            GPIO.output(ina, self.HIGH)
            GPIO.output(inb, self.LOW)
        
        elif stat == self.BACKWORD:
            GPIO.output(ina, self.LOW)
            GPIO.output(inb, self.HIGH)
    
        elif stat == self.STOP:
            GPIO.output(ina, self.LOW)
            GPIO.output(inb, self.LOW)
            
    def SetMotor(self, ch, speed, stat):
        if ch == self.CH1:
            self.SetMotorContorl(self.pwmA, self.IN1, self.IN2, speed, stat)
        else:
            self.SetMotorContorl(self.pwmB, self.IN3, self.IN4, speed, stat)

    def Go(self):  
        self.SetMotor(self.CH1, 100, self.FORWARD)
        self.SetMotor(self.CH2, 100, self.FORWARD)
    
    def Back(self): 
        self.SetMotor(self.CH1, 100, self.BACKWORD)
        self.SetMotor(self.CH2, 100, self.BACKWORD)
        
    def Stop(self): 
        self.SetMotor(self.CH1, 80, self.STOP)
        self.SetMotor(self.CH2, 80, self.STOP)

    def Right(self): 
        self.SetMotor(self.CH1, 100, self.FORWARD)
        self.SetMotor(self.CH2, 100, self.BACKWORD)

    def Left(self): 
        self.SetMotor(self.CH1, 100, self.BACKWORD)
        self.SetMotor(self.CH2, 100, self.FORWARD)
 
    def Clean(self):
        GPIO.cleanup()

if __name__ == '__main__':
    wheel = MotorDriver(26, 0, 19, 13, 6, 5)
    
    while True:
        n = input(":")
        if n == '1':
            wheel.Go()
        elif n == '2':
            wheel.Right()
        elif n == '3':
            wheel.Left()
        elif n == '0':
            wheel.Stop()
        else:
            wheel.Clean()
            del wheel
            break
            