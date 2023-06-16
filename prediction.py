import joblib

def predict(data):
    clf = joblib.load("ridge_model")
    return clf.predict(data)