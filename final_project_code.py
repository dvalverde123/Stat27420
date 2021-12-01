import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, log_loss
from xgboost import XGBClassifier, XGBRegressor
import sklearn
import os
from austen_plots.AustenPlot import AustenPlot
from doubleml import DoubleMLData
from doubleml import DoubleMLPLR
import doubleml as dml

crime_data_2002_2010 = pd.read_csv("2002-2010_crimes.csv")
crime_data_small_2002_2010 = crime_data_2002_2010[["Year", "Community Area"]]
grouped_2002_2010 = crime_data_small_2002_2010.groupby(["Year", "Community Area"]).size()
grouped_2002_2010.to_csv('2002_2010_crime_groupings.csv')
crime_data_2011_2020 = pd.read_csv("2011-2020_crimes.csv")
crime_data_small_2011_2020 = crime_data_2011_2020[["Year", "Community Area"]]
grouped_2011_2020 = crime_data_small_2011_2020.groupby(["Year", "Community Area"]).size()
grouped_2011_2020.to_csv('2011_2020_crime_groupings.csv')

# create nuisance functions 

RANDOM_SEED = 12194292
np.random.seed(RANDOM_SEED)

# random forest model 
def create_random_forest_Q():
    return RandomForestRegressor(random_state = RANDOM_SEED, n_estimators = 500)

random_forest_Q = create_random_forest_Q()

# gradient boosting model 
def create_xgb_Q():
    return XGBClassifier()

# OLS
def create_ols_Q():
    return OLS

# LASSO
def create_LASSO():
    return LASSO

X_w_treatment = confounders.copy()
X_w_treatment['treatment'] = treatment

X_train, X_trest, Y_train, Y_test = train_test_split(X_w_treatment, outcome, test_size = 0.2)
random_forest_Q.fit(X_train, Y_train)
xgb_Q_fit(X_train, Y_train)
ols_Q_fit(X_train, Y_train)
LASSO_Q_fit(X_train, Y_train)

fit_MSE = mean_squared_error(Y_pred, Y_test)
baseline_MSE = mean_squared_erro(y_train.mean()*np.ones_like(Y_test), Y_test)
