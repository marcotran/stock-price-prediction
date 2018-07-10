import math
import random
import numpy as np
from flask import jsonify
from sklearn import preprocessing

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

    X = np.array(mlData.drop(['label'],1))
    X = preprocessing.scale(X)
    X_data = X[-dataLength:]
    X = X[:-dataLength]
    data = mlData[-dataLength:]
    mlData = mlData[:-dataLength]
    y = np.array(mlData['label'])
    return X, y, X_data, data

def packageData(data, prediction, accuracy):
    data = data[['Adj. Close']]
    data = data.rename(columns={'Adj. Close':'EOD'})
    data['prediction'] = prediction[:]
    data = data.to_json(orient='table')
    return jsonify(data)