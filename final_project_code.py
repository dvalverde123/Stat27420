import numpy as np
import pandas as pd
import scipy as sp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, log_loss
import sklearn
import os
import matplotlib as plt
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import pathlib
from austen_plots.AustenPlot import AustenPlot
#from doubleml import DoubleMLData
#from doubleml import DoubleMLPLR


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
    outcome = crime_data[str(outcome_year) + "_cr_per_100k"]
    confounders = crime_data[["Birth Rate", "Pop_" + year, "Assault (Homicide)", 
        "Below Poverty Level", "Per Capita Income", "Unemployment", "HARDSHIP INDEX"]]

    return treatment, outcome, confounders
    

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)

# choose model for the conditional expected outcome

treatment, outcome, confounders = define_variables("2004")

X_w_treatment = confounders.copy()
X_w_treatment['treatment'] = treatment
X_train, X_test, Y_train, Y_test = train_test_split(X_w_treatment, outcome, test_size=0.2)

# random_forest
random_forest_Q = RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=500, max_depth=None)
random_forest_Q.fit(X_train, Y_train)
RF_Y_Pred = random_forest_Q.predict(X_test)

test_mse_rf = mean_squared_error(RF_Y_Pred, Y_test)
print(f"Test MSE of random forest model {test_mse_rf}") 

# gradient boosting 
xgb_Q = XGBRegressor().fit(X_train, Y_train)
XGB_Y_Pred = xgb_Q.predict(X_test)

test_mse_xgb = mean_squared_error(XGB_Y_Pred, Y_test)
print(f"Test MSE of gradient boosting model {test_mse_xgb}") 

# linear regression 
regression_Q = LinearRegression().fit(X_train, Y_train)
regression_Y_Pred = regression_Q.predict(X_test)

test_mse_lr = mean_squared_error(regression_Y_Pred, Y_test)
print(f"Test MSE of linear regression model {test_mse_lr}") 

# k nearest neighbors 
knn_Q = KNeighborsRegressor().fit(X_train, Y_train)
knn_Y_Pred = knn_Q.predict(X_test)

test_mse_knn = mean_squared_error(knn_Y_Pred, Y_test)
print(f"Test MSE of k-nearest neighbors model {test_mse_knn}")

# baseline MSE

baseline_mse_knn = mean_squared_error(Y_train.mean()*np.ones_like(Y_test), Y_test)
print(f"Test MSE of no-covariate model {baseline_mse_knn}")

# XGB gives lowest MSE, so we choose XGB model for conditional expected outcome

def make_Q_model():
    return LinearRegression()

Q_model = make_Q_model()


# diff in diff data cleaning 

# 2004 school closings 
# before period

"""
2004_closed = crime_data['Treatment_2004'].is_equalto(1)
crime_data['Treatment_2004'] = 2004_closed 

# after treatment
compact_df=crime_data[~crime_data['2004_closed']]
crime_rate_changes = crime_data['2009_cr_per_100k'].values
compact_df['2009-2004'] = crime_data['2009_cr_per_100k'] - crime_data['2004_cr_per_100k']

# format for ideal ATT processing 
compact_df = compact_df.reset_index()

outcome = compact_df['2009-2004']
treatment = compact_df['2004_closed']
confounders = compact_df[['all of them']]

# 2013 school closings 

# original data
2013_closed = crime_data['Treatment_2013'].is_equalto(1)
crime_data['Treatment_2004'] = 2013_closed 

# after treatment
compact_df=crime_data[~crime_data['2013_closed']]
crime_rate_changes = crime_data['2018_cr_per_100k'].values
compact_df['2018-2013'] = crime_data['2018_cr_per_100k'] - crime_data['2013_cr_per_100k']

# format for ideal ATT processing 
compact_df = compact_df.reset_index()

outcome = compact_df['2018-2013']
treatment = compact_df['2013_closed']
confounders = compact_df[['all of them']]

"""

# propensity scores model

X_train, X_test, A_train, A_test = train_test_split(confounders, treatment)

# random forest 
random_forest_g = RandomForestClassifier(n_estimators=100, max_depth=2)
random_forest_g.fit(X_train, A_train)
RF_A_Pred = random_forest_g.predict_proba(X_test)[:,1]
test_cross_entropy = log_loss(A_test, RF_A_Pred)
print(f"Test CE of random forest model {test_cross_entropy}") 

# gradient boosting 
xgb_g = XGBClassifier().fit(X_train, A_train)
XGB_A_Pred = xgb_g.predict_proba(X_test)
test_cross_entropy = log_loss(A_test, XGB_A_Pred)
print(f"Test CE of gradient boosting model {test_cross_entropy}") 

# logistic regression 
regression_g = LogisticRegressionCV(max_iter=1000).fit(X_train, A_train)
regression_A_Pred = regression_g.predict(X_test)
test_cross_entropy = log_loss(A_test, regression_A_Pred)
print(f"Test CE of logistic regression model {test_cross_entropy}") 

# baseline CE
baseline_cross_entropy = log_loss(A_test, A_train.mean()*np.ones_like(A_test))
print(f"Test CE of no-covariate model {baseline_cross_entropy}")

def make_g_model():
    return RandomForestClassifier(n_estimators=100, max_depth=2)

g_model = make_g_model()


# Use cross fitting to get predicted outcomes and propensity scores for each unit
def treatment_k_fold_fit_predict(make_model, X:pd.DataFrame, A:np.array, n_splits:int):
    predictions = np.full_like(A, np.nan, dtype=float)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    for train_index, test_index in kf.split(X,A):
        X_train = X.loc[train_index]
        A_train = A.loc[train_index]
        g = make_model()
        g.fit(X_train, A_train)

        predictions[test_index] = g.predict_proba(X.loc[test_index])[:,1]

    assert np.isnan(predictions).sum() == 0
    return predictions

def outcome_k_fold_fit_predict(make_model, X:pd.DataFrame, y:np.array, A:np.array, n_splits:int, output_type:str):
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

g = treatment_k_fold_fit_predict(make_g_model, X=confounders, A=treatment, n_splits=7)

Q0, Q1 = outcome_k_fold_fit_predict(make_Q_model, X=confounders, y=outcome, A=treatment, n_splits=10, output_type='continuous')

data_nuisance_estimates = pd.DataFrame({'g': g, 'Q0': Q0, 'Q1': Q1, 'A': treatment, 'Y': outcome})
data_nuisance_estimates.head()


# Double ML estimator for ATT
def att_aiptw(Q0, Q1, g, A, Y, prob_t=None):
    if prob_t is None:
        prob_t = A.mean()

    tau_hat = (A*(Y-Q0) - (1-A)*(g/(1-g))*(Y-Q0)).mean()/ prob_t

    scores = (A*(Y-Q0) - (1-A)*(g/(1-g))*(Y-Q0) - tau_hat*A) / prob_t
    n = Y.shape[0]
    std_hat = np.std(scores) / np.sqrt(n)
    
    return tau_hat, std_hat

tau_hat, std_hat = att_aiptw(**data_nuisance_estimates)
print(f"The ATT estimate is {tau_hat} pm {1.96*std_hat}")


# Double ML estimator for ATE 
def ate_aiptw(Q0, Q1, g, A, Y, prob_t=None):
    tau_hat = (Q1-Q0 + A*(Y-Q1)/g - (1-A) * (Y-Q0)/(1-g)).mean()
    
    scores = Q1 - Q0 + A*(Y-Q1)/g - (1-A) * (Y-Q0)/(1-g) - tau_hat
    n = Y.shape[0]
    std_hat = np.std(scores) / np.sqrt(n)
    
    return tau_hat, std_hat

in_treated = data_nuisance_estimates['A']==1
treated_estimates = data_nuisance_estimates[in_treated]
tau_hat, std_hat = ate_aiptw(**treated_estimates)
print(f"The ATE estimate is {tau_hat} pm {1.96*std_hat}")

# address overlap issues here 
g = data_nuisance_estimates['g']
in_overlap_popluation = (g < 0.90)
overlap_data_and_nuisance = data_nuisance_estimates[in_overlap_popluation]
tau_hat, std_hat = att_aiptw(**overlap_data_and_nuisance)
print(f"The ATT estimate with restricted population is {tau_hat} pm {1.96*std_hat}")

# point estimate without covariate correction
outcome[treatment==1].mean()-outcome[treatment==0].mean()

"""

# Sensitivity Analysis

# create covariate groups 

"""

year = "2004"

covariate_groups = {
    'economic': ["Per Capita Income", "HARDSHIP INDEX", "Below Poverty Level"],
    'population': ["Birth Rate", "Pop_" + year, "Assault (Homicide)"]}

# for each covariate group, refit models without using that group 
nuisance_estimates = {}
for group, covariates in covariate_groups.items():
    remaining_confounders = confounders.drop(columns=covariates)

    g = treatment_k_fold_fit_predict(make_g_model, X=remaining_confounders, A=treatment, n_splits=5)
    Q0, Q1 = outcome_k_fold_fit_predict(make_Q_model, X=remaining_confounders, y=outcome, A=treatment, n_splits=5, output_type="continuous")
    data_nuisance_estimates = pd.DataFrame(({'g': g, 'Q0': Q0, 'Q1': Q1, 'A': treatment, 'Y': outcome}))
    nuisance_estimates[group] = data_nuisance_estimates

data_nuisance_path = 'data_nuisance_estimates.csv'
covariate_dir_path = 'covariates/'

def convert_to_austen_format(nuisance_estimate_df: pd.DataFrame):
    austen_df = pd.DataFrame()
    austen_df['y']=nuisance_estimate_df['Y']
    austen_df['t']=nuisance_estimate_df['A']
    austen_df['g']=nuisance_estimate_df['g']
    A = nuisance_estimate_df['A']
    austen_df['Q']=A*nuisance_estimate_df['Q1'] + (1-A)*nuisance_estimate_df['Q0']

    return austen_df

austen_data_nuisance = convert_to_austen_format(data_nuisance_estimates)
austen_data_nuisance.to_csv(data_nuisance_path, index=False)

pathlib.Path(covariate_dir_path).mkdir(exist_ok=True)
for group, nuisance_estimate in nuisance_estimates.items():
    austen_nuisance_estimate = convert_to_austen_format(nuisance_estimate)
    austen_nuisance_estimate.to_csv(os.path.join(covariate_dir_path+'.csv'), index=False)

# Austen Plots 
target_bias = 3000


ap = AustenPlot(data_nuisance_path, covariate_dir_path)
p, plot_coords, variable_coords = ap.fit(bias=target_bias)
p


