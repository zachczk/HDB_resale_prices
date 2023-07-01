import numpy as np
import joblib

def predict(data):
    clf = joblib.load("xgb_model.pkl")
    return clf.predict(data)