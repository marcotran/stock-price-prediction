# standard py libs
import math
import random
import os

# project dependencies
import quandl
import numpy as np
import sklearn.linear_model as lin
from flask import Flask, request, send_from_directory
from sklearn import preprocessing, cross_validation

# project modules
from utils import FormatForModel, PackageData
from regression import LinearRegression, ARIMARegression, LSTMRegression

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('public','index.html')
@app.route('/index.html')
def index2():
    return send_from_directory('public','index.html')
@app.route('/scripts/app.js')
def index_app_script():
    return send_from_directory('public/scripts', 'app.js')
@app.route('/scripts/lib/moment.js')
def index_moment_script():
    return send_from_directory('public/scripts/lib', 'moment.js')
@app.route('/scripts/lib/angular-moment.min.js')
def index_angular_moment_script():
    return send_from_directory('public/scripts/lib', 'angular-moment.min.js')
@app.route('/styles/main.css')
def index_styles():
    return send_from_directory('public/styles', 'main.css')
@app.route('/data/stocks.json')
def index_json_stock_data():
    return send_from_directory('public/data', 'stocks.json')
@app.route('/data/chart_config.json')
def index_json_chart_data():
    return send_from_directory('public/data', 'chart_config.json')

@app.route('/getstockdata/')
def getStockData():
    # stock = request.args.get('stock', default="IBM", type=None)
    # method = request.args.get('method', default="1", type=None)
    stock = "IBM"
    quandl.ApiConfig.api_key = "qWcicxSctVxrP9PhyneG"
    allData = quandl.get('WIKI/'+stock)
    dataLength = 251
    allDataLength = len(allData)
    firstDataElem = math.floor(random.random()*(allDataLength-dataLength))
    mlData = allData[0:firstDataElem+dataLength]

    mlData = FormatForModel(mlData)

    forecast_col = 'Adj. Close'
    forecast_out = int(math.ceil(0.12*dataLength))

    mlData['label'] = mlData[forecast_col].shift(-forecast_out)
    mlData.dropna(inplace=True)

    X = np.array(mlData.drop(['label'],1))
    X = preprocessing.scale(X)
    X_data = X[-dataLength:]
    X = X[:-dataLength]
    data = mlData[-dataLength:]
    mlData = mlData[:-dataLength]
    y = np.array(mlData['label'])

    prediction, accuracy = LinearRegression(X, y, X_data)
    return PackageData(data, prediction, accuracy)

if __name__ == '__main__':
    if os.getenv('ENV', 'dev') == 'prod':
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        app.run(debug=True)