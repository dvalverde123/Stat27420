import numpy as np
import pandas as pd
import scipy as sp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, log_loss
import sklearn
import os
import matplotlib as plt
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
#from austen_plots.AustenPlot import AustenPlot
#from doubleml import DoubleMLData
#from doubleml import DoubleMLPLR
#import doubleml as dml


crime_data = pd.read_csv("CLEANED_DATA.csv")

def define_variables(year):
    """
    Define treatment and outcome for a given year

    Input:
        year (string): year of school closing
    
    Return:
        treatment
        outcome
        confounders
    """

    outcome_year = int(year) + 5

    treatment = crime_data["Treatment_" + year]
    outcome = crime_data["Crime_" + str(outcome_year)]
    confounders = crime_data[["Birth Rate", "Pop_" + str(outcome_year), "Assault (Homicide)", 
        "Below Poverty Level", "Per Capita Income", "Unemployment", "HARDSHIP INDEX"]]

    return treatment, outcome, confounders
    

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)

# specify nuisance function models 

# choose model for the conditional expected outcome

# gradient boosting model 
def create_xgb_Q():
    return XGBClassifier

# linear regression model 
# https://scikit-learn.org/stable/modules/linear_model.html
def create_linear_regression_Q():
    reg = LinearRegression()
    return reg


# k-nearest_neighbors
# https://scikit-learn.org/stable/modules/neighbors.html
def create_k_nearest_neighbors_Q():
    return 'HI'

# random_forest

random_forest_Q = RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=500, max_depth=None)
treatment, outcome, confounders = define_variables("2004")

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

# cross fitting 

def treatment_k_fold_fit_predict(make_model, X:pd.DataFrame, A:np.array, n_splits:int):
    predictions = np.full_like(A, np.nan, dtype=float)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    for train_indx, test_index in kf.split(X,A):
        X_train = X.loc[train_index]
        A_train = A.loc[train_index]
        g = make_model()
        g.fit(X_train, A_train)

        predictions[test_index] = g.predict_proba(X.loc[test_idex])[:,1]

    assert np.isnan(predictions).sum() == 0
    return predictions

def outcome_k_fold_fit_predict:
    predictions0 = np.full_like(A, np.nan, dtype=float)
    predictions1 = np.full_like(y, np.nan, dtype=float)
    if output_type == 'binary':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    elif output_type == 'continuous':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    X_w_treatment = X.copy()
    X_w_treatment["A"] = A

    X0 = X_w_treatment.copy()
    X0["A"] = 0
    X1 = X_w_treatment.copy()
    X1["A"] = 1

    for train_index, test_index in kf.split(X_w_treatment, y):
        X_train = X_w_treatment.loc[train_index]
        y_train = y.loc[train_index]
        q = make_model()
        q.fit(X_train, y_train)

        if output_type == 'binary':
            predictions0[test_index] = q.predict_proba(X0.loc[test_index])[:, 1]
            predictions1[test_index] = q.predict_proba(X1.loc[test_index])[:, 1]
        elif output_type == 'continuous':
            predictions0[test_index] = q.predict(X0.loc[test_index])
            predictions1[test_index] = q.predict(X1.loc[test_index])

    assert np.isnan(predictions0).sum() == 0
    assert np.isnan(predictions1).sum() == 0
    return predictions0, predictions1

g = treatment_k_fold_fit_predict(make_g_model, X=confounders, A=treatment, n_splits=10)

Q0,Q1=outcome_k_fold_fit_predict(make_Q_model, X=confounders, y=outcome, A=treatment, n_splits=10, output_type='continuous')

data_nuisance_estimates = pd.DataFrame({'g': g, 'Q0': Q0, 'Q1': Q1, 'A': treatment, 'Y': outcome})
data_nuisance_estimates.head()



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

in_treated = data_nuisance_estimates['A']==1
treated_estimates = data_nuisance_estimates[in_treated]
tau_hat, std_hat = ate_aiptw(**treated_estimates)

print(f"The estimate is {tau_hat} pm {1.96*std_hat}")

# address overlap issues here 

# Double ML Library Estimation Procedure 



# Differences in Differences Estimation 

tau_hat, std_hat = att_aiptw(**data_nuisance_estimates)

# point estimate without covariate correction

outcome[treatment==1].mean()-outcome[treatment==0.mean()]

# Sensitivity Analysis

# create covariate groups 

covariate_groups = {
    'economic':
    'population':
    'age':
    'health':
}

# refit models for each covariate group 

nuisance_estimates = {}

for group, covariates in covariate_groups.items():
    remaining_confounders = confounders.drop(columns=covariates)

    g = treatment_k_fold_fit_predict(make_g_model, X=remaining_confounders)
    Q0, Q1 = outcome_k_fold_fit_predict(make_Q_model, X=remaining_confounders)
    data_nuisance_estimates = pd.DataFrame(({'g': g, 'Q0': Q0, 'Q1': Q1, 'A': treatment, 'Y': outcome}))
    nuisance_estimates[group] = data_nuisance_estimates

data_nuisance_path = 'data_nuisance_estimates.csv'
covariate_direct_path = 'covariates/'

def convert_to_austen_format(nuisance_estimate_df: pd.DataFrame):
    austen_df = pd.DataFrame()
    austen_df['y']=nuisance_estimate_df['Y']
    austen_df['t']=nuisance_estimate_df['A']
    austen_df['g']=nuisance_estimate_df['g']
    A = nuisance_estimate_df['A']
    austen_df['Q']=A*nuisance_estimate_df['Q1'] + (1-A)*nuisance_estimate_df['Q0']

austen_data_nuisance = convert_to_austen_format(data_nuisance_estimate)
austen_data_and_nuisance.to_csv(data_and_nuisance_apth, index=False)

pathlib.Path(covariate_dir_path).mkdir(exist_ok=True)
for group, nuisance_estimate in nuisance_estimates.items():
    austen_nuisance_estimate = convert_to_austen_format(nuisance_estimate)
    austen_nuisance_estimate.to_csv(os.path.join(covariate_dir_path+'.csv'), index=False)

# Austen Plots 
target_bias = 15.0 


austen_plot = AustenPlot(data_nuisance, covariate_path)
p, plot_cooredinates, variable_coordinates = austen_plot.fit(bias = target_bias)
p

"""

