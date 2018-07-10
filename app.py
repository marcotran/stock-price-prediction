# standard py libs
import os

# project dependencies
import quandl
from flask import Flask, request, send_from_directory

# project modules
from utils import processData, packageData 
from regression import LinearRegression, BayesianRidge, RidgeRegression, SupportVectorMachine, ARIMARegression, LSTMRegression

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
    apiData = quandl.get('WIKI/'+stock)
    
    X, y, X_data, data = processData(apiData)

    prediction, accuracy = None, None

    if method == 1:
        prediction, accuracy = LinearRegression(X, y, X_data)
    elif method == 2:
        prediction, accuracy = BayesianRidge(X, y, X_data)
    elif method == 3:
        prediction, accuracy = RidgeRegression(X, y, X_data)
    elif method == 4:
        prediction, accuracy = SupportVectorMachine(X, y, X_data)
    elif method == 5:
        prediction, accuracy = ARIMARegression(X, y, X_data)
    elif method == 6:
        prediction, accuracy = LSTMRegression(X, y, X_data)
    else:
        pass
    
    print(accuracy)
    return packageData(data, prediction, accuracy)

if __name__ == '__main__':
    if os.getenv('ENV', 'dev') == 'prod':
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        app.run(debug=True)