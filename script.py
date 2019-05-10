import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
from sklearn import datasets

# creating instance of the class
app=Flask(__name__)

iris = datasets.load_iris()
print(iris.data)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,4)
    loaded_model = pickle.load(open("classifier.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        # convert string inputs to integer values
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
        print(result)
        prediction = str(iris.target_names[result])

        return render_template("result.html",  prediction=prediction)
