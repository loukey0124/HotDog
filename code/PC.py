from threading import Thread
from flask import Flask

import tflite_object_detect

app = Flask(__name__)

predict = tflite_object_detect.Predict()

@app.route('/Activate')
def Activate():
    predict.DogActivate()
    return '200'

@app.route('/getlocation')
def Location():
    return predict.GetDogLocaiton()

@app.route('/Deactivate')
def Deactivate():
    predict.DogDeactivate()
    return '200'

@app.route('/return/Activate')
def returnActivate():
    predict.CatActivate()
    return '200'

@app.route('/return/Deactivate')
def returnDeactivate():
    predict.CatDeactivate()
    return '200'

@app.route('/getcatlocation')
def CatLocation():
    return predict.GetCatLocaiton()
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)