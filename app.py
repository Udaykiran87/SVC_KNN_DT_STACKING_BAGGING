from flask import Flask, render_template, request

import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pandas_profiling import ProfileReport
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import Parallel, delayed
import time
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LassoCV, ElasticNet , ElasticNetCV,LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

app = Flask(__name__)

@app.route('/')
def form():
    return render_template('form.html')


@app.route('/data/', methods=['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"

    if request.method == "POST":
        frontal_axis_reading = float(request.form['frontal_axis_reading(g)'])
        vertical_axis_reading = float(request.form['vertical_axis_reading(g)'])
        lateral_axis_reading = float(request.form['lateral_axis_reading(g)'])
        if request.form.get("Predict_Activity_using_SVC"):
            file = './notebook/svc_model'
        elif request.form.get("Predict_Activity_using_KNN"):
            file = './notebook/knn_model'
        elif request.form.get("Predict_Activity_using_Decision_Tree"):
            file = './notebook/dt_model'
        elif request.form.get("Predict_Activity_using_Random_Forest"):
            file = './notebook/rf_model'
        elif request.form.get("Predict_Activity_using_Stacking"):
            file = './notebook/stack_model'
        saved_model = pickle.load(open(file, 'rb'))
        prediction = saved_model.predict([[frontal_axis_reading,vertical_axis_reading,lateral_axis_reading]])
        print('prediction is', prediction)
        return render_template('results.html', prediction=prediction)


@app.route('/profie_report/', methods=['POST', 'GET'])
def profie_report():
    return render_template('har_profiling.html')


if __name__ == '__main__':
    app.run(host='localhost', port=5000)