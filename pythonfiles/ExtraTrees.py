# THIS FILE IS TO TRAIN THE MODEL ONLY


# IMPORTS
print("<<< IMPORTING MODULES >>>")
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
import joblib
print("<<< MODULES ARE IMPORTED >>>")

print("<<< LOADING TRAIN AND TEST DATA >>>")
# TRAIN AND TEST DATA
X = pd.read_csv("../train-test_data/X.csv")
y = pd.read_csv("../train-test_data/y.csv").Class
print("<<< TRAIN AND TEST DATA ARE LOADED >>>")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

print("<<< TRAINING MODEL >>>")
# MODEL TRAINING
etc = XGBClassifier()
etc.fit(X_train, y_train)
print("<<< MODEL IS TRAINED >>>")

print("<<< SAVING MODEL >>>")
# SAVE MODEL
joblib.dump(etc, 'etc.sav')
print("<<< MODEL SAVED >>>")
print("<<< DONE >>>")
