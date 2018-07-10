from sklearn import cross_validation, svm
import sklearn.linear_model as lin
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

def DatasetSplit(X, y):
    return cross_validation.train_test_split(X, y, test_size=0.3)

def LinearRegression(X, y, X_data):
    X_train, X_test, y_train, y_test = DatasetSplit(X, y)
    clf = lin.LinearRegression()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    prediction = clf.predict(X_data)
    return prediction, accuracy

def BayesianRidge(X, y, X_data):
    X_train, X_test, y_train, y_test = DatasetSplit(X, y)
    clf = lin.BayesianRidge()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    prediction = clf.predict(X_data)
    return prediction, accuracy

def RidgeRegression(X, y, X_data):
    X_train, X_test, y_train, y_test = DatasetSplit(X, y)
    clf = lin.Ridge()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    prediction = clf.predict(X_data)
    return prediction, accuracy

def SupportVectorMachine(X, y, X_data):
    X_train, X_test, y_train, y_test = DatasetSplit(X, y)
    clf = svm.SVR()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    prediction = clf.predict(X_data)
    return prediction, accuracy

def ARIMARegression(X, y, X_data):
    X_train, X_test, y_train, y_test = DatasetSplit(X, y)

def LSTMRegression(X, y, X_data):
    X_train, X_test, y_train, y_test = DatasetSplit(X, y)
    
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

    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))