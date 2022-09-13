# THIS FILE IS TO TRAIN THE MODEL ONLY


# IMPORTS
import numpy as np
import pandas as pd 
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# TRAIN AND TEST DATA
X_train = np.array(pd.read_csv("../train-test_data/X_train.csv"))
X_test = np.array(pd.read_csv("../train-test_data/X_test.csv"))
y_train = np.array(pd.read_csv("../train-test_data/y_train.csv"))
y_test = np.array(pd.read_csv("../train-test_data/y_test.csv"))

# MODEL TRAINING
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# SAVE MODEL
joblib.dump(rfc, '../models/rfc.sav')
