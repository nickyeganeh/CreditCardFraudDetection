# THIS FILE IS TO TRAIN THE MODEL ONLY


# IMPORTS
print("<<< IMPORTING MODULES >>>")
import numpy as np
import pandas as pd 
from sklearn.ensemble import AdaBoostClassifier
import joblib
print("<<< MODULES ARE IMPORTED >>>")


print("<<< LOADING TRAIN AND TEST DATA >>>")
# TRAIN AND TEST DATA
X_train = pd.read_csv("../train-test_data/X_train.csv")
y_train = pd.read_csv("../train-test_data/y_train.csv").Class
print("<<< TRAIN AND TEST DATA ARE LOADED >>>")


print("<<< TRAINING MODEL >>>")
# MODEL TRAINING
abc = AdaBoostClassifier(n_estimators = 100, random_state = 1)
abc.fit(X_train, y_train)
print("<<< MODEL IS TRAINED >>>")


print("<<< SAVING MODEL >>>")
# SAVE MODEL
joblib.dump(abc, 'abc.sav')
print("<<< MODEL SAVED >>>")
print("<<< DONE >>>")
