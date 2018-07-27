import math
import random
import numpy as np
from flask import jsonify
from sklearn import preprocessing
from sklearn import cross_validations

def DatasetSplit(X, y):
    return cross_validation.train_test_split(X, y, test_size=0.1)

def formatForModel(dataArray):
    dataArray = dataArray[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
    dataArray['HL_PCT'] = (dataArray['Adj. High'] - dataArray['Adj. Close']) / dataArray['Adj. Close'] * 100.0
    dataArray['PCT_change'] = (dataArray['Adj. Close'] - dataArray['Adj. Open']) / dataArray['Adj. Open'] * 100.0
    dataArray = dataArray[['Adj. Close', 'HL_PCT', 'PCT_change','Adj. Volume']]
    dataArray.fillna(-99999, inplace=True)
    return dataArray

def processData(apiData):
    dataLength = 251
    allDataLength = len(apiData)
    firstDataElem = math.floor(random.random()*(allDataLength-dataLength))
    mlData = apiData[0:firstDataElem+dataLength]

    mlData = formatForModel(mlData)

    forecast_col = 'Adj. Close'
    forecast_out = int(math.ceil(0.12*dataLength))

    mlData['label'] = mlData[forecast_col].shift(-forecast_out)
    mlData.dropna(inplace=True)

    # get all the columns except the ouput label
    X = np.array(mlData.drop(['label'], 1))
    # preprocess data
    X = preprocessing.scale(X)
    # separate some of the data to be predicted

    X_arima = mlData[['label']]

    X_arima_predict = X_arima[-dataLength:]
    X_arima_model = X_arima[:-dataLength]

    X_predict = X[-dataLength:]
    # separate the rest of the data for the model to be trained on
    X_model = X[:-dataLength]
    # get the output labels of the data to be predicted
    y_actual = mlData[-dataLength:]
    y_actual = y_actual[['Adj. Close']]
    mlData = mlData[:-dataLength]
    y_model = np.array(mlData['label'])

    return X_model, y_model, X_predict, y_actual, X_arima_model, X_arima_predict

def packageData(data, prediction, accuracy):
    data = data.rename(columns={'Adj. Close':'actual'})
    data['prediction'] = prediction[:]
    data = data.to_json(orient='table')
    return jsonify(data)