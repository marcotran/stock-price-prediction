from math import sqrt

import quandl
import numpy as np
from sklearn import cross_validation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from pandas import Series, DataFrame, concat
from matplotlib import pyplot

from utils import processData

# DATA PREPROCESSING FUNCTIONS

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data) # make a dataframe of the data
    # make a list of all dataframes with successive lag values 
    columns = [df.shift(i) for i in range(1, lag+1)]
    # append the original dataframe to the list
    columns.append(df)
    # concatenate the dataframes in the list into one
    df = concat(columns, axis=1)
    # fill NaN values with 0
    df.fillna(0, inplace=True)
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# LSTM MODEL FUNCTIONS

def fit_model(train, batch_size, epochs, neurons):
    
    # preprocess the training set
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    # create and define the sequential model
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    
    # compile the model using loss as MSE and ADAM optimizer
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # fit the model on the training set for set no. of epochs 
    # controlling how the state of the model changes with each epoch
    for i in range(epochs):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    
    return model

def forecast(model, batch_size, X):
    
    # retrieve and reshape the input as req.
    X = X.reshape(1, 1, len(X))
    
    # predict the next value
    yhat = model.predict(X, batch_size=batch_size)
    
    return yhat[0, 0]

def LSTMRegression(train_prep, test_prep):

    # stock = "AAPL"

    # quandl.ApiConfig.api_key = "M46EXcBvFPiHWDrdAFnY"   #"qWcicxSctVxrP9PhyneG"
    # apiData = quandl.get('WIKI/' + stock)

    # _, _, _, _, train_prep, test_prep = processData(apiData)

    # transform data to be stationary
    train_raw, test_raw = train_prep.values, test_prep.values
    train_diff, test_diff = difference(train_raw, 1), difference(test_raw, 1)
    
    # transform data to be supervised learning
    train_supervised, test_supervised = timeseries_to_supervised(train_diff, 1), timeseries_to_supervised(test_diff, 1)
    train, test = train_supervised.values, test_supervised.values
    
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    
    # fit the model with 4 LSTM neurons for batch of 1 and 3000 epochs 
    lstm_model = fit_model(train_scaled, 1, 500, 4)

    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)

    # seed the state by making a prediction on all samples 
    # in the training dataset so that  the internal state 
    # be set up ready to forecast the next time step
    lstm_model.predict(train_reshaped, batch_size=1)
    
    # walk-forward validation on the test data
    predictions = list()

    # iteratively predict on each element in test set
    for i in range(len(test_scaled)):
        
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        
        # make the prediction
        yhat = forecast(lstm_model, 1, X)
        
        # invert scaling and differencing
        yhat = invert_scale(scaler, X, yhat)
        yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)

        # store forecast
        predictions.append(yhat)
        
        # print result
        expected = raw_values[len(train) + i + 1]
        # print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

    # report model performance
    rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
    return predictions, rmse
    # print('Test RMSE: %.3f' % rmse)

    # # line plot of observed vs predicted
    # pyplot.plot(test_raw)
    # pyplot.plot(predictions)
    # pyplot.show()

# LSTMRegression()