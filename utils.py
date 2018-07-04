from flask import jsonify

def FormatForModel(dataArray):
    dataArray = dataArray[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
    dataArray['HL_PCT'] = (dataArray['Adj. High'] - dataArray['Adj. Close']) / dataArray['Adj. Close'] * 100.0
    dataArray['PCT_change'] = (dataArray['Adj. Close'] - dataArray['Adj. Open']) / dataArray['Adj. Open'] * 100.0
    dataArray = dataArray[['Adj. Close', 'HL_PCT', 'PCT_change','Adj. Volume']]
    dataArray.fillna(-99999, inplace=True)
    return dataArray

def PackageData(data, prediction, accuracy):
    data = data[['Adj. Close']]
    data = data.rename(columns={'Adj. Close':'EOD'})
    data['prediction'] = prediction[:]
    data = data.to_json(orient='table')
    return jsonify(data)