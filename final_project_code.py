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
baseline_MSE = mean_squared_error(y_train.mean()*np.ones_like(Y_test), Y_test)

# propensity scores model 

def create_g():
    return Logistic Regression(max_iter=1000)
    return RandomForestClassifier(n_estimators=100, max_depth=5)
g_model = create_g()

X_train, X_test, A_train, A_test = train_test_split(confounders, treatment)
g_model.fit(X_train, A_train)
A_pred = g_model.predict_proba(X_test)[:,1]

test_cross_entropy = log_loss(A_test, A_pred)
baseline_cross_entropy = log_loss(A_test, A_Train.mean()*np.ones_like(A_test))

# could do cross fitting here? Idt it will work tho 

# Double ML estimator for ATT
def att_aiptw(Q0, Q1, g, A, Y, prob_t=None):
    if prob_t is None:
        prob_t = A.mean()
    tau_hat = (A*(Y-Q0) - (1-A)*(g/(1-g))*(Y-Q0) - tau_hat*A) / prob_t
    n = Y.shape[0]
    std_hat = np.std(scores) / np.sqrt(n)
    
    return tau_hat, std_hat

# Double ML estimator for ATE 
def ate_aiptw(Q0, Q1, g, A, Y, prob_t=None):
    tau_hat = (Q1-Q0 + A*(Y-Q1)/g - (1-A) * (Y-Q0)/(1-g)).mean()
    
    scores = Q1 - Q0 + A*(Y-Q1)/g - (1-A) * (Y-Q0)(1-g) - tau_hat
    n = Y.shape[0]
    std_hat = np.std(scores) / np.sqrt(n)
    
    return tau_hat, std_hat

# Double ML Library Estimation Procedure 

# Differences in Differences Estimation 

# Austen Plots 
target_bias = 15.0 
covariates = {}
estimates_for_nuisance = {}
for group, covariates in covariates.items():
    narrowed_confounders = confounders.drop(columns = covariates)
    
g = treatment_k_fold_fit_and_predict(create_g_model, X = narrowed_confounders, A = treatment, n_splits = 5)
Q0, Q1 = outcome_k_fold_fit_and_predict()

data_nuisance_estimates = pd.DataFrame(({'g': g, 'Q0': Q0, 'Q1': Q1, 'A': treatment, 'Y': outcome}))
nuisance_estimates[group] = data_nuisance_estimates
austen_plot = AustenPlot(data_nuisance, covariate_path)
p, plot_cooredinates, variable_coordinates = austen_plot.fit(bias = target_bias)


