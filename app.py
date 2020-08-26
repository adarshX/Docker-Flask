## App without any frontend UI
from flask import Flask,request
import pandas as pd 
#import numpy as np
import pickle



## creating app
app =Flask(__name__)

## classifier file
pickle_file = open("RF_Classifier.pkl",'rb')
classifier = pickle.load(pickle_file)


## creating routes
@app.route("/")
def hello():
    return "Hello ady"

## fucntion for predcition --> we need four features according to our model
@app.route('/predict' , methods = ['GET']) # by default its in 'GET' request mode
def predict_note():
    var  = request.args.get('Variance')
    skew = request.args.get('Skewness')
    curt = request.args.get('Curtosis')
    ent = request.args.get('Entropy')
    pred = classifier.predict([[var , skew , curt , ent]])
    output = "The predciton value for given bank note details : " + str(pred) + "." 
    return output

## sample GET data : 
#  ?Variance=5.1321&Skewness=-0.031048&Curtosis=0.32616&Entropy=1.11510    -> Test data : ans = 0
#  ?Variance=-1.2943&Skewness=2.6735&Curtosis=-0.84085&Entropy=-2.03230   -> train data : ans = 1
# 			

@app.route('/predict_file' , methods = ['POST'])
def  predict_note_file():
    file_name = request.files.get("file")  ## file variable will get csv file
    df_test = pd.read_csv(file_name)
    pred = classifier.predict(df_test)
    output = "The predciton value for given bank note in a csv file : " + str(list(pred)) + "." 
    return output


## running app
if __name__== "__main__":
    app.run()