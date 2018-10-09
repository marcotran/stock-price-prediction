import numpy as np
import pandas as pd
from flask import jsonify
from sklearn import preprocessing
import json

import warnings
warnings.filterwarnings('ignore')

def process_data(stock_data):
    pred_data = 365
    forecast_column = 'Adj Close'
    dates = np.array(stock_data['Date'].values)
    X = np.array(stock_data[forecast_column].values)[1:]
    y = np.array(stock_data[forecast_column].shift(1))[1:]
    X = preprocessing.scale(X)

    return dates[-pred_data:], X[:-pred_data].reshape(-1, 1), X[-pred_data:].reshape(-1, 1), y[:-pred_data], y[-pred_data:]

def package_data(dates, y_predict, prediction, result):
    data = {'date': dates[:], 'actual': y_predict[:], 'prediction': prediction[:]}
    df = pd.DataFrame(data=data)
    data = df.to_json(orient='table')
    data = json.loads(data)
    data['result'] = result
    return jsonify(data)
