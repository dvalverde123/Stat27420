import numpy as np
import pandas as pd
import scipy as sp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, log_loss
import sklearn
import os
import matplotlib as plt
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import pathlib
from austen_plots.AustenPlot import AustenPlot


crime_data = pd.read_csv("CLEANED_DATA.csv")

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)


def find_estimators(year):
    """
    Finds ATT and ATE estimator under diff in diff for given year of school closings
    
    Input:
        year (string): year of school closing
    Returns:
        Prints ATT and ATE and performs Austen plots
    """

    treatment, outcome, confounders = define_variables(year)

    # finds best Q and g nuisance functions
    Q_model = find_Q_model(treatment, outcome, confounders)
    g_model = find_g_model(treatment, outcome, confounders)

    g = treatment_k_fold_fit_predict(g_model, X=confounders, A=treatment, n_splits=7)

    Q0, Q1 = outcome_k_fold_fit_predict(Q_model, X=confounders, y=outcome, \
        A=treatment, n_splits=10, output_type='continuous')

    data_nuisance_estimates = pd.DataFrame({'g': g, 'Q0': Q0, 'Q1': Q1, 'A': treatment, 'Y': outcome})
    data_nuisance_estimates.head()

    # finds ATT estimate
    tau_hat_att, std_hat_att = att_aiptw(**data_nuisance_estimates)
    print(f"The ATT estimate is {tau_hat_att} pm {1.96*std_hat_att}")

    # finds ATE estimate
    in_treated = data_nuisance_estimates['A']==1
    treated_estimates = data_nuisance_estimates[in_treated]
    tau_hat_ate, std_hat_ate = ate_aiptw(**treated_estimates)
    print(f"The ATE estimate is {tau_hat_ate} pm {1.96*std_hat_ate}")

    # address overlap issues here 
    new_g = data_nuisance_estimates['g']
    in_overlap_popluation = (new_g < 0.90)
    overlap_data_and_nuisance = data_nuisance_estimates[in_overlap_popluation]
    tau_hat, std_hat = att_aiptw(**overlap_data_and_nuisance)
    print(f"The ATT estimate with restricted population is {tau_hat} pm {1.96*std_hat}")

    # point estimate without covariate correction
    outcome[treatment==1].mean()-outcome[treatment==0].mean()

    # perform sensitivity analysis and plot
    sensitivity_analysis(treatment, outcome, confounders, year, g_model, Q_model)

    # test conditional parallel trends
    expected_treated = []
    year_treated = []
    expected_untreated = []
    year_untreated = []

    for i in range(2002,2013):
        year = str(i)
        treatment_pt, outcome_pt, confounders = define_variables(year)
        Q0, Q1 = outcome_k_fold_fit_predict(Q_model, X=confounders, y=outcome, \
            A=treatment, n_splits=10, output_type='continuous')
        Q = Q1-Q0
        if treatment == 1:
            expected_treated.append(Q)
            year_treated.append(i)
        else:
            expected_untreated.append(Q)
            year_untreated.append(i)

        plt.plot(year_treated, expected_treated, label = "treated")
        plt.plot(year_untreated, expected_untreated, label = "untreated")
        plt.legend()
        plt.show


    data_nuisance_estimates = pd.DataFrame({'Q0': Q0, 'Q1': Q1, 'A': treatment, 'Y': outcome})


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
        "Below Poverty Level", "Per Capita Income", "Unemployment", "Males_15_25", 
        "MED_AGE", "WHITE", "HISP", "BLACK", "ASIAN", "NOT_ENGLISH"]]

    return treatment, outcome, confounders


def find_Q_model(treatment, outcome, confounders):
    """
    Finds model for conditional expected outcome that minimizes MSE, comparing
    random forest, gradient boosting, linear regression, k nearest neighbors,
    and baseline
    Inputs:
        treatment
        outcome
        confounders
    Returns:
        Prints out MSE under each model
    """

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
    baseline_mse = mean_squared_error(Y_train.mean()*np.ones_like(Y_test), Y_test)
    print(f"Test MSE of no-covariate model {baseline_mse}")

    if min(test_mse_rf, test_mse_xgb, test_mse_lr, test_mse_knn) == test_mse_rf:
        return RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=500, max_depth=None)
    elif min(test_mse_rf, test_mse_xgb, test_mse_lr, test_mse_knn) == test_mse_xgb:
        return XGBRegressor()
    elif min(test_mse_rf, test_mse_xgb, test_mse_lr, test_mse_knn) == test_mse_lr:
        return LinearRegression()
    else:
        return KNeighborsRegressor()


def find_g_model(treatment, outcome, confounders):
    """
    Finds model for propensity score that minimizes cross entropy, comparing
    random forest, gradient boosting, logistic regression, and baseline
    Inputs:
        treatment
        outcome
        confounders
    Returns:
        Prints out CE under each model
    """

    X_train, X_test, A_train, A_test = train_test_split(confounders, treatment, test_size=0.2)

    # random forest 
    random_forest_g = RandomForestClassifier(n_estimators=100, max_depth=2)
    random_forest_g.fit(X_train, A_train)
    RF_A_Pred = random_forest_g.predict_proba(X_test)[:,1]
    test_ce_rf = log_loss(A_test, RF_A_Pred)
    print(f"Test CE of random forest model {test_ce_rf}") 

    # gradient boosting 
    xgb_g = XGBClassifier().fit(X_train, A_train)
    XGB_A_Pred = xgb_g.predict_proba(X_test)
    test_ce_xgb = log_loss(A_test, XGB_A_Pred)
    print(f"Test CE of gradient boosting model {test_ce_xgb}") 

    # logistic regression 
    regression_g = LogisticRegressionCV(solver = "liblinear", max_iter=1000).fit(X_train, A_train)
    regression_A_Pred = regression_g.predict_proba(X_test)
    test_ce_lr = log_loss(A_test, regression_A_Pred)
    print(f"Test CE of logistic regression model {test_ce_lr}") 

    # k nearest neighbors 
    knn_Q = KNeighborsClassifier().fit(X_train, A_train)
    knn_A_Pred = knn_Q.predict_proba(X_test)
    test_ce_knn = log_loss(A_test, knn_A_Pred)
    print(f"Test CE of k-nearest neighbors model {test_ce_knn}")

    # baseline CE
    baseline_cross_entropy = log_loss(A_test, A_train.mean()*np.ones_like(A_test))
    print(f"Test CE of no-covariate model {baseline_cross_entropy}")

    if min(test_ce_rf, test_ce_xgb, test_ce_lr, test_ce_knn) == test_ce_rf:
        return RandomForestClassifier(n_estimators=100, max_depth=2)
    elif min(test_ce_rf, test_ce_xgb, test_ce_lr) == test_ce_xgb:
        return XGBClassifier()
    elif min(test_ce_rf, test_ce_xgb, test_ce_lr, test_ce_knn) == test_ce_lr:
        return LogisticRegressionCV(max_iter=1000)
    else:
        return KNeighborsClassifier()


# Use cross fitting to get predicted outcomes and propensity scores for each unit
def treatment_k_fold_fit_predict(g_model, X:pd.DataFrame, A:np.array, n_splits:int):
    predictions = np.full_like(A, np.nan, dtype=float)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    for train_index, test_index in kf.split(X,A):
        X_train = X.loc[train_index]
        A_train = A.loc[train_index]
        g_model.fit(X_train, A_train)

        predictions[test_index] = g_model.predict_proba(X.loc[test_index])[:,1]

    assert np.isnan(predictions).sum() == 0
    return predictions

def outcome_k_fold_fit_predict(q_model, X:pd.DataFrame, y:np.array, A:np.array, n_splits:int, output_type:str):
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
        q_model.fit(X_train, y_train)

        if output_type == 'binary':
            predictions0[test_index] = q_model.predict_proba(X0.loc[test_index])[:, 1]
            predictions1[test_index] = q_model.predict_proba(X1.loc[test_index])[:, 1]
        elif output_type == 'continuous':
            predictions0[test_index] = q_model.predict(X0.loc[test_index])
            predictions1[test_index] = q_model.predict(X1.loc[test_index])

    assert np.isnan(predictions0).sum() == 0
    assert np.isnan(predictions1).sum() == 0
    return predictions0, predictions1


def att_aiptw(Q0, Q1, g, A, Y, prob_t=None):
    """
    Finds Double ML estimator for ATT
    """

    if prob_t is None:
        prob_t = A.mean()

    tau_hat = (A*(Y-Q0) - (1-A)*(g/(1-g))*(Y-Q0)).mean()/ prob_t

    scores = (A*(Y-Q0) - (1-A)*(g/(1-g))*(Y-Q0) - tau_hat*A) / prob_t
    n = Y.shape[0]
    std_hat = np.std(scores) / np.sqrt(n)
    
    return tau_hat, std_hat


def ate_aiptw(Q0, Q1, g, A, Y, prob_t=None):
    """
    Finds Double ML estimator for ATE
    """
    tau_hat = (Q1-Q0 + A*(Y-Q1)/g - (1-A) * (Y-Q0)/(1-g)).mean()
    
    scores = Q1 - Q0 + A*(Y-Q1)/g - (1-A) * (Y-Q0)/(1-g) - tau_hat
    n = Y.shape[0]
    std_hat = np.std(scores) / np.sqrt(n)
    
    return tau_hat, std_hat


def sensitivity_analysis(treatment, outcome, confounders, year, g_model, Q_model):
    """
    Perform Austen Plots sensitivity analysis
    Input:
        treatment
        outcome
        confounders
        year (string)
    Return:
        plot
    """

    covariate_groups = {
        'economic': ["Per Capita Income", "Below Poverty Level"],
        'population': ["Birth Rate", "Pop_" + year, "Assault (Homicide)"], 
        'demographics': ["Males_15_25", "MED_AGE", "WHITE", "HISP", "BLACK", "ASIAN"],
        'language': "NOT_ENGLISH"}

    # for each covariate group, refit models without using that group 
    nuisance_estimates = {}
    for group, covs in covariate_groups.items():
        remaining_confounders = confounders.drop(columns=covs)

        g = treatment_k_fold_fit_predict(g_model, X=remaining_confounders, A=treatment, n_splits=5)
        Q0, Q1 = outcome_k_fold_fit_predict(Q_model, X=remaining_confounders, \
            y=outcome, A=treatment, n_splits=5, output_type="continuous")
        data_nuisance_estimates = pd.DataFrame(({'g': g, 'Q0': Q0, 'Q1': Q1, 'A': treatment, 'Y': outcome}))
        nuisance_estimates[group] = data_nuisance_estimates

    data_nuisance_path = 'data_nuisance_estimates.csv'
    covariate_dir_path = 'covariates/'

    austen_data_nuisance = convert_to_austen_format(data_nuisance_estimates)
    austen_data_nuisance.to_csv(data_nuisance_path, index=False)

    pathlib.Path(covariate_dir_path).mkdir(exist_ok=True)
    for group, nuisance_estimate in nuisance_estimates.items():
        austen_nuisance_estimate = convert_to_austen_format(nuisance_estimate)
        austen_nuisance_estimate.to_csv(os.path.join(covariate_dir_path, group+'.csv'), index=False)

    # Austen Plots 
    target_bias = 2500

    ap = AustenPlot(data_nuisance_path, covariate_dir_path)
    p, plot_coords, variable_coords = ap.fit(bias=target_bias)

    return p


def convert_to_austen_format(nuisance_estimate_df: pd.DataFrame):
    austen_df = pd.DataFrame()
    austen_df['y']=nuisance_estimate_df['Y']
    austen_df['t']=nuisance_estimate_df['A']
    austen_df['g']=nuisance_estimate_df['g']
    A = nuisance_estimate_df['A']
    austen_df['Q']=A*nuisance_estimate_df['Q1'] + (1-A)*nuisance_estimate_df['Q0']

    return austen_df
