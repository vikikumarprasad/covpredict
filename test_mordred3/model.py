import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem

from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, PassiveAggressiveRegressor, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from ngboost import NGBRegressor

#import warnings
#from sklearn.exceptions import ConvergenceWarning
#warnings.filterwarnings("ignore", category=ConvergenceWarning)

regressor_configs = {
    'XGBRegressor': {
        #'n_estimators': [10, 50, 100, 200, 500, 1000],
        'learning_rate': [1e-1, 1, 10, 100, 1000],
        #'max_depth': [1, 3, 5, 7, 9],
        #'subsample': [0.5, 0.75, 1.0],
        #'colsample_bytree': [0.5, 0.75, 1.0],
        #'min_child_weight': [1, 5, 10]
    }
}

def display_results(model_name, X_train, X_test, y_train, y_test, y_scaler):
    best_model = joblib.load(f'{model_name}_best_model.pkl')
    #y_scaler = joblib.load(y_scaler)
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    #y_train_pred = y_scaler.inverse_transform(y_train_pred.reshape(-1, 1))
    #y_test_pred = y_scaler.inverse_transform(y_test_pred.reshape(-1, 1))
    #y_train = y_scaler.inverse_transform(y_train)
    #y_test = y_scaler.inverse_transform(y_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f'{model_name} Model Performance:')
    print(f'Train set: R2= {train_r2}, MAE = {train_mae}, RMSE = {train_rmse}')
    print(f'Test set: R2 = {test_r2}, MAE = {test_mae}, RMSE = {test_rmse}')

    #y_train_pred_file = (f'y_train_{model_name}.csv') 
    #y_test_pred_file = (f'y_test_{model_name}.csv')
    #y_train_pred.to_csv(y_train_pred_file)
    #y_test_pred.to_csv(y_test_pred_file)

    sns.set_style("white")
    plt.figure(figsize= (5,5))
    plt.xlabel("DFT Barrier Height $(kcal/mol)$" , fontsize = 14)
    plt.ylabel("ML Barrier Height $(kcal/mol)$" , fontsize = 14)
    plt.plot(y_train , y_train_pred , 'o' , color = "red" , markersize = 1 )
    plt.plot(y_test , y_test_pred , 'o' , color = "blue" , markersize = 1 )
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{model_name}_plot.pdf",bbox_inches='tight')
    return

def fit_and_evaluate(model_name, model_class, data):
    X_train, X_test, y_train, y_test = data

    print(f"Fitting {model_name}...")
    model = model_class()
    model.fit(X_train, y_train)
    joblib.dump(model, f'{model_name}_model.pkl')
    print(f"Model fitting complete for {model_name}.")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    #print(f"{model_name} Model Performance:")
    #print(f"Train MAE: {train_mae}, Train RMSE: {train_rmse}")
    #print(f"Test MAE: {test_mae}, Test RMSE: {test_rmse}")

    return model

def run_grid_search(model_name, model_class, data):
    X_train, X_test, y_train, y_test = data
    param_grid = regressor_configs[model_name]
    print(f'Starting GridSearchCV for {model_name}...')
    grid_search = GridSearchCV(model_class(), param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
    #grid_search = GridSearchCV(model_class(), param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f'{model_name}_best_model.pkl')
    print(f"GridSearchCV complete for {model_name}.")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score (MAE): {-grid_search.best_score_}")

    return grid_search

def main():
    X_train = pd.read_csv('X_train-mordred3.csv')
    X_test = pd.read_csv('X_test-mordred3.csv')
    y_train = pd.read_csv('y_train-mordred3.csv')
    y_test = pd.read_csv('y_test-mordred3.csv')
    y_scaler = 'y_scaler.pkl'

    models = {
        #'Ridge': Ridge,
        #'ElasticNet': ElasticNet,
        #'BayesianRidge': BayesianRidge,
        #'PassiveAggressiveRegressor': PassiveAggressiveRegressor,
        #'HuberRegressor': HuberRegressor,
        #'KNeighborsRegressor': KNeighborsRegressor,
        #'RandomForestRegressor': RandomForestRegressor,
        #'GradientBoostingRegressor': GradientBoostingRegressor,
        #'ExtraTreesRegressor': ExtraTreesRegressor,
        #'DecisionTreeRegressor': DecisionTreeRegressor,
        #'SVR': SVR,
        #'KernelRidge': KernelRidge,
        #'GaussianProcessRegressor': GaussianProcessRegressor,
        'XGBRegressor': XGBRegressor,
        #'LGBMRegressor': LGBMRegressor,
        #'CatBoostRegressor': CatBoostRegressor,
        #'AdaBoostRegressor': AdaBoostRegressor,
        #'NGBRegressor': NGBRegressor
    }

    for model_name, model_class in models.items():
        print(f'Running model fitting for {model_name}')
        #fit_and_evaluate(model_name, model_class, (X_train, X_test, y_train, y_test))
        grid_search = run_grid_search(model_name, model_class, (X_train, X_test, y_train, y_test))
        print(f'Best configuration for {model_name}: {grid_search.best_params_}')
        display_results(model_name, X_train, X_test, y_train, y_test, y_scaler)

if __name__ == '__main__':
    main()
