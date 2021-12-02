import numpy as np
import pandas as pd
import scipy as sp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, log_loss
import sklearn
import os
import matplotlib as plt
#from xgboost import XGBClassifier, XGBRegressor
#from austen_plots.AustenPlot import AustenPlot
#from doubleml import DoubleMLData
#from doubleml import DoubleMLPLR
#import doubleml as dml


# not really sure which columns you wanna use for confounders and which is outcome

crime_data = pd.read_csv("CLEANED_DATA.csv")
confounders = crime_data["Birth Rate", "Pop_" + str(outcome_year), "Assault (Homicide",
"Below Poverty Level", "Per Capita Income", "Unemployment", "HARDSHIP INDEX"] # and whatever other confounders we decide

def define_variables(year):
    """
    Define treatment and outcome for a given year

    Input:
        year (string): year of school closing
    
    Return:
        treatment
        outcome
    """

    outcome_year = int(year) + 5

    treatment = crime_data["Treatment_" + year]
    outcome = crime_data["Crime_" + str(outcome_year)]

    return treatment, outcome
    

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)

# specify nuisance function models 

# choose model for the conditional expected outcome

# random forest model that returns sklearn model for later use in k-folding
def create_random_forest_Q():
    return RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=500, max_depth=None)

# gradient boosting model 
#def create_xgb_Q():
    #return XGBClassifier

# linear regression model 
# https://scikit-learn.org/stable/modules/linear_model.html
#def create_linear_regression_Q():
    #return "hi"


# k-nearest_neighbors
# https://scikit-learn.org/stable/modules/neighbors.html
#def create_k_nearest_neighbors_Q():
    #return 'HI'


random_forest_Q = create_random_forest_Q()
outcome, treatment = define_variables("2004")

X_w_treatment = confounders.copy()
X_w_treatment['treatment'] = treatment

X_train, X_test, Y_train, Y_test = train_test_split(X_w_treatment, outcome, test_size=0.2)
random_forest_Q.fit(X_train, Y_train)
Y_Pred = random_forest_Q.predict(X_test)


test_mse = mean_squared_error(Y_Pred, Y_test)
print(f"Test MSE of fit model {test_mse}") 
baseline_mse=mean_squared_error(Y_train.mean()*np.ones_like(Y_test), Y_test)
print(f"Test MSE of no-covariate model {baseline_mse}")

# gradient boosting 
xgb_Q = create_xgb_Q.fit(X_train, Y_train)
XGB_Y_Pred = xgb_Q.predict(X_test)
test_mse = mean_squared_error(Y_Pred, Y_test)
print(f"Test MSE of fit model {test_mse}") 
baseline_mse=mean_squared_error(Y_train.mean()*np.ones_like(Y_test), Y_test)
print(f"Test MSE of no-covariate model {baseline_mse}")

# linear regression 
regression_Q = create_linear_regression_Q.fit(X_train, Y_train)
regression_Y_Pred = regression_Q.predict(X_test)
test_mse = mean_squared_error(Y_Pred, Y_test)
print(f"Test MSE of fit model {test_mse}") 
baseline_mse=mean_squared_error(Y_train.mean()*np.ones_like(Y_test), Y_test)
print(f"Test MSE of no-covariate model {baseline_mse}")

# k nearest neighbords 


# model evaluation
# cross validation sci kit learn package? 

"""
# propensity scores model 

def create_g():
    return Logistic_Regression(max_iter=1000)
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

"""

