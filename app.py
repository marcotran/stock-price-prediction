# standard py libs
import os

# project dependencies
import quandl
import pandas as pd
from flask import Flask, request, send_from_directory

# project modules
from utils import processData, packageData 
from regression import *

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
@app.route('/data/methods.json')
def index_json_method_data():
    return send_from_directory('public/data', 'methods.json')
@app.route('/data/chart_config.json')
def index_json_chart_data():
    return send_from_directory('public/data', 'chart_config.json')

@app.route('/getstockdata/')
def getStockData():
    stock = request.args.get('stock', default="IBM")
    method = int(request.args.get('method', default="1"))
    print(stock)
    print(method)

    quandl.ApiConfig.api_key = "M46EXcBvFPiHWDrdAFnY"   #"qWcicxSctVxrP9PhyneG"
    apiData = quandl.get('WIKI/' + stock)
    
    X_model, y_model, X_predict, y_actual, arima_model, arima_predict = processData(apiData)

    prediction, accuracy = None, None

    if method == 1:
        prediction, accuracy = LinearRegression(X_model, y_model, X_predict)
    elif method == 2:
        prediction, accuracy = BayesianRidge(X_model, y_model, X_predict)
    elif method == 3:
        prediction, accuracy = RidgeRegression(X_model, y_model, X_predict)
    elif method == 4:
        prediction, accuracy = SupportVectorMachine(X_model, y_model, X_predict)
    elif method == 5:
        prediction, accuracy = ARIMARegression(arima_model, arima_predict)
    elif method == 6:
        prediction, accuracy = LSTMRegression(X_model, y_model, X_predict)
    elif method == 7:
        prediction, accuracy = ARDRegression(X_model, y_model, X_predict)
    elif method == 8:
        prediction, accuracy = ElasticNet(X_model, y_model, X_predict)
    else:
        pass

    print(accuracy)
    return packageData(y_actual, prediction, accuracy)

if __name__ == '__main__':
    if os.getenv('ENV', 'dev') == 'prod':
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        app.run(debug=True)