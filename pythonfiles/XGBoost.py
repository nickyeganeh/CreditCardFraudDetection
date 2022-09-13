# THIS FILE IS TO TRAIN THE MODEL ONLY


# IMPORTS
print("<<< IMPORTING MODULES >>>")
import numpy as np
import pandas as pd 
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import joblib
print("<<< MODULES ARE IMPORTED >>>")

print("<<< LOADING TRAIN AND TEST DATA >>>")
# TRAIN AND TEST DATA
X_train = pd.read_csv("../train-test_data/X_train.csv")
X_test = pd.read_csv("../train-test_data/X_test.csv")
y_train = pd.read_csv("../train-test_data/y_train.csv")
y_test = pd.read_csv("../train-test_data/y_test.csv")
print("<<< TRAIN AND TEST DATA ARE LOADED >>>")

print("<<< TRAINING MODEL >>>")
# MODEL TRAINING
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
print("<<< MODEL IS TRAINED >>>")

print("<<< SAVING MODEL >>>")
# SAVE MODEL
joblib.dump(xgbc, '../models/xgbc.sav')
print("<<< MODEL SAVED >>>")
print("<<< DONE >>>")
