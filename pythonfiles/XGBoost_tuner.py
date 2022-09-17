import numpy as np
import pandas as pd 
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

print("<<< LOADING TRAINING DATA >>>")
# TRAIN AND TEST DATA
pipeline_X_train = pd.read_csv("../train-test_data/pipeline_X_train.csv")
pipeline_y_train = pd.read_csv("../train-test_data/pipline_y_train.csv").Class
print("<<< TRAINING DATA LOADED >>>")

print("<<< PERFORMING GRIDSEARCH >>>")
model = Pipeline([
        ('sampling', SMOTE()),
        ('classification', XGBClassifier())
    ])

params = {"n_estimators" : [100, 200, 300],
		"learning_rate" : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
		"scale_pos_neg" : [1, 199007/357],
		"max_depth" : [ 3, 4, 5, 6, 8, 10, 12, 15],
		"min_child_weight" : [ 1, 3, 5, 7],
		"min_split_loss" : [ 0.0, 0.1, 0.2 , 0.3, 0.4],
		"tree_method" : ["hist", "auto"]}

grid = GridSearchCV(estimator = model, param_grid = params, scoring = "f1_weighted", n_jobs = 4)
grid.fit(pipeline_X_train, pipeline_y_train)
print("<<< GRIDSEARCH COMPLETE >>>")

print("<<< SAVING GRID >>>")
# SAVE MODEL
joblib.dump(grid, 'xgbc_grid.pk1')
print("<<< GRID SAVED >>>")
print("<<< DONE >>>")
