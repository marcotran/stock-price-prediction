from sklearn import cross_validation, svm
import sklearn.linear_model as lin
from statsmodels.tsa import arima_model
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import numpy as np

def DatasetSplit(X, y):
    return cross_validation.train_test_split(X, y, test_size=0.1)

def LinearRegression(X_model, y_model, X_predict):
    X_train, X_test, y_train, y_test = DatasetSplit(X_model, y_model)
    clf = lin.LinearRegression()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    prediction = clf.predict(X_predict)
    return prediction, accuracy

def BayesianRidge(X_model, y_model, X_predict):
    X_train, X_test, y_train, y_test = DatasetSplit(X_model, y_model)
    clf = lin.BayesianRidge()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    prediction = clf.predict(X_predict)
    return prediction, accuracy

def RidgeRegression(X_model, y_model, X_predict):
    X_train, X_test, y_train, y_test = DatasetSplit(X_model, y_model)
    clf = lin.Ridge()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    prediction = clf.predict(X_predict)
    return prediction, accuracy

def SupportVectorMachine(X_model, y_model, X_predict):
    X_train, X_test, y_train, y_test = DatasetSplit(X_model, y_model)
    clf = svm.SVR()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    prediction = clf.predict(X_predict)
    return prediction, accuracy

def ARIMARegression(model, predict):
    series = model.squeeze()
    test = predict.squeeze()
    history = [x for x in series]
    predictions = list()
    for t in range(len(test)):
        model = arima_model.ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat[0])
        obs = test[t]
        history.append(obs)
    error = mean_squared_error(test, predictions)
    predictions = np.array(predictions)
    accuracy = error
    print('ARIMA Test MSE: %.3f' % error)
    return predictions, accuracy

def LSTMRegression(X_model, y_model, X_predict):
    X_train, X_test, y_train, y_test = DatasetSplit(X_model, y_model)

    input_dim = None
    output_dim = None 
    return_sequences = None
    
    model = Sequential()
    model.add(LSTM(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences))
    model.add(Dropout(0.2))
    model.add(LSTM(
        128, 
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ["accuracy"])
    model.fit(X=X_train, y=y_train, epochs=20, batch_size=50, verbose=1)

    preds = model.evaluate(X=X_test, y=y_test, verbose  = 1)

    loss = preds[0]
    accuracy = preds[0]

    prediction = model.predict(X_predict)

    print()
    print ("Loss = " + str(loss))
    print ("Test Accuracy = " + str(accuracy))
    print(model.predict(X_data))

    return prediction, accuracy

def ARDRegression(X_model, y_model, X_predict):
    X_train, X_test, y_train, y_test = DatasetSplit(X_model, y_model)
    clf = lin.ARDRegression()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    prediction = clf.predict(X_predict)
    return prediction, accuracy

def ElasticNet(X_model, y_model, X_predict):
    X_train, X_test, y_train, y_test = DatasetSplit(X_model, y_model)
    clf = lin.ElasticNet()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    prediction = clf.predict(X_predict)
    return prediction, accuracy