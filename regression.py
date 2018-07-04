from sklearn import cross_validation
import sklearn.linear_model as lin

def DatasetSplit(X, y):
    return cross_validation.train_test_split(X, y, test_size=0.3)

def LinearRegression(X, y, X_data):
    X_train, X_test, y_train, y_test = DatasetSplit(X, y)
    clf = lin.LinearRegression()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    prediction = clf.predict(X_data)
    return prediction, accuracy

def ARIMARegression(X, y, X_data):
    X_train, X_test, y_train, y_test = DatasetSplit(X, y)

def LSTMRegression(X, y, X_data):
    X_train, X_test, y_train, y_test = DatasetSplit(X, y)