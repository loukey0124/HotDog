# -*- coding: utf-8 -*-

from threading import Thread, Event
from flask import Flask, render_template, Response
import requests
import cv2
from time import sleep

import RPi.GPIO as GPIO
import atexit

from ball import Ball
from servo import Servo
from motordriver import MotorDriver
from audio import Speaker
from sounddetect import SoundDetect

#========== init ==========
app = Flask(__name__, template_folder='template')

url = 'http://192.168.0.43:8080/'
event = Event()

def ExitApp():
    GPIO.cleanup()

atexit.register(ExitApp)

#========== main menu ==========

#도크에서 나가는 버튼
@app.route('/start')
def Start():
    servo1 = Servo(23)
    servo2 = Servo(14)
    
    servo1.Activate(9)
    servo2.Activate(9)
    sleep(1)
    servo1.Activate(3)
    servo2.Activate(3)
    sleep(1)
    
    del servo1
    del servo2
    
    wheel.Back()
    sleep(5)
    wheel.Right()
    sleep(7.5)
    wheel.Stop()
    return 'done'

#도크로 복귀하는 버튼
@app.route('/return')
def returnDock():
    event.clear()
    
    try:
        requests.get(url=url + 'return/Activate', timeout=10)
    except:
        return 'SERVER NOT STARTED'
    
    print("return start")
    Thread(target=GetLocation, args=('getcatlocation',), daemon=True).start()
    return "returnDock"

#========== camera stream ==========

camera = cv2.VideoCapture(0)  # use 0 for web camera

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if (not success):
            break

        else:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream')
def index():
    return render_template('index.html')

#========== play mode ==========

ball = Ball(1)
wheel = MotorDriver(26, 0, 19, 13, 6, 5)
foodServo = Servo(18)
foodServo.Activate(5)
sleep(0.5)
foodServo.Activate(0)

@app.route('/playmode/activate')
def playmode_activate():
    ball.Activate()
    return '200'

@app.route('/playmode/deactivate')
def playmode_deactivate():
    ball.Deactivate()
    return '200'

# 전진
@app.route('/btnfront')
def btnfront():
    wheel.Go()
    return "Front"

@app.route('/btnback')
def btnback():
    wheel.Back()
    return "back"

#우회전
@app.route('/btnright')
def btnright():
    wheel.Right()
    return "Right"

#좌회전
@app.route('/btnleft')
def btnleft():
    wheel.Left()
    return "Left"

#멈춤
@app.route('/stop')
def btnstop():
    wheel.Stop()
    return "stop"

#간식주는 버튼
@app.route('/feed')
def Feed():
    foodServo.Activate(7)
    sleep(0.3)
    foodServo.Activate(5)
    sleep(0.3)
    foodServo.Activate(0)
    return "feed"

speaker = Speaker()
@app.route('/playsound')
def PlaySound():
    speaker.Play()
    return '200'

#========== 관리모드 ==========

soundclassifier = SoundDetect()

def GetLocation(url2):
    loc = "None"
    while True:
        if event.is_set():
            return
        
        loc = requests.get(url=url + url2).text
        print(loc)
        if loc == 'go':
            print("go")
            wheel.Go()
        elif loc == 'right':
            print("right")
            wheel.Right()
        elif loc == 'left':
            print("left")
            wheel.Left()
        elif loc == 'stop':
            print("stop")
            wheel.Stop()
        elif loc == 'END':
            requests.get(url=url + 'return/Deactivate')
            event.set()
            wheel.Go()
            sleep(3)
            wheel.Stop()
            print("Done")
            return "200"
        else:
            print("not detect")
            wheel.Left()
        
        sleep(0.1)

@app.route('/caremode/activate')
def CareModeActivate():
    event.clear()
    
    try:
        requests.get(url=url + 'Activate', timeout=10)
    except:
        return 'SERVER NOT STARTED'
    
    print("carestart")
    Thread(target=GetLocation, args=('getlocation',), daemon=True).start()
    soundclassifier.Activate()
    return '200'
    
@app.route('/caremode/deactivate')
def CareModeDeactivate():
    wheel.Stop()
    event.set()
    soundclassifier.Deactivate()
    requests.get(url=url + 'Deactivate')
    
    return '200'

if (__name__ == '__main__'):
    app.run(host='0.0.0.0', port=8080)