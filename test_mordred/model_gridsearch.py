import os
import pandas as pd
import numpy as np
import joblib
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

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Setup the hyperparameter search space
regressor_configs = {
    'Ridge': {
        'alpha': [1e-1, 1, 10, 50],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    },
    'ElasticNet': {
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        'l1_ratio': [0, 0.25, 0.5, 0.75, 1],
        'selection': ['cyclic', 'random']
    },
    'BayesianRidge': {
        'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3],
        'alpha_2': [1e-6, 1e-5, 1e-4, 1e-3],
        'lambda_1': [1e-6, 1e-5, 1e-4, 1e-3],
        'lambda_2': [1e-6, 1e-5, 1e-4, 1e-3]
    },
    'PassiveAggressiveRegressor': {
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
        'epsilon': [0, 0.25, 0.5, 0.75, 1]
    },
    'HuberRegressor': {
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        'epsilon': [1.35, 1.5, 1.75, 2]
    },
    'KNeighborsRegressor': {
        'n_neighbors': list(range(1, 31)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'RandomForestRegressor': {
        'n_estimators': [10, 50, 100, 200, 500, 1000],
        'max_depth': [1, 5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'GradientBoostingRegressor': {
        'n_estimators': [10, 50, 100, 200, 500, 1000],
        'learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
        'max_depth': [1, 3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'ExtraTreesRegressor': {
        'n_estimators': [10, 50, 100, 200, 500, 1000],
        'max_depth': [1, 5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'DecisionTreeRegressor': {
        'max_depth': [1, 5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None]
    },
    'SVR': {
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        'epsilon': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['auto', 'scale'],
        'degree': [2, 3, 4],
        'coef0': [0, 0.5, 1],
        'max_iter': [100, 1000, 5000]
    },
    'KernelRidge': {
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        'kernel': ['linear', 'rbf', 'laplacian', 'polynomial', 'exponential', 'chi2', 'sigmoid'],
        'gamma': ['auto', 'scale'],
        'degree': [2, 3, 4],
        'coef0': [0, 0.5, 1]
    },
    'GaussianProcessRegressor': {
        'alpha': [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1],
        'length_scale': [1e-2, 1e-1, 1, 10],
        'n_restarts_optimizer': [10, 20, 50],
        #kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        #kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        'kernel': ['rbf', 'matern', 'rationalquadratic']
    },
    'XGBRegressor': {
        'n_estimators': [10, 50, 100, 200, 500, 1000],
        'learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
        'max_depth': [1, 3, 5, 7, 9],
        'subsample': [0.5, 0.75, 1.0],
        'colsample_bytree': [0.5, 0.75, 1.0],
        'min_child_weight': [1, 5, 10]
    },
    'LGBMRegressor': {
        'n_estimators': [10, 50, 100, 200, 500, 1000],
        'learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
        'max_depth': [1, 3, 5, 7, 9],
        'num_leaves': [31, 63, 127],
        'feature_fraction': [0.5, 0.75, 1.0],
        'bagging_fraction': [0.5, 0.75, 1.0]
    },
    'CatBoostRegressor': {
        'iterations': [10, 50, 100, 200, 500, 1000],
        'learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
        'depth': [1, 3, 5, 7, 9],
        'l2_leaf_reg': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        'border_count': [32, 64, 128]
    },
    'AdaBoostRegressor': {
        'n_estimators': [10, 50, 100, 200, 500, 1000],
        'learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
        'loss': ['linear', 'square', 'exponential']
    },
    'NGBRegressor': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
        'minibatch_frac': [0.5, 0.75, 1.0],
        'natural_gradient': [True, False]
    }
}

# Function to run hyperparameter tuning with GridSearchCV
def run_grid_search(model_name, model_class, data):
    X_train, X_test, y_train, y_test = data
    param_grid = regressor_configs[model_name]

    print(f'Starting GridSearchCV for {model_name}...')
    
    def logging_callback(status):
        print(f"Evaluating parameters: {status.parameters}")
        print(f"Results so far: {status.results}")

    grid_search = GridSearchCV(model_class(), param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
    #grid_search = GridSearchCV(model_class(), param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f'{model_name}_best_model.pkl')

    print(f"GridSearchCV complete for {model_name}.")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score (MAE): {-grid_search.best_score_}")

    return grid_search

# Function to generate and save performance plots
#def plot_performance(grid_search, model_name):
#    results = pd.DataFrame(grid_search.cv_results_)
#    results['mean_test_scores'] = -results['mean_test_score'] # Convert to positive MAE
#    param_alpha = results['param_alpha']
#    plt.plot(param_alpha, mean_test_scores)
#    plt.xlabel('Alpha')
#    plt.ylabel('Mean Test Score')
#    plt.title(f'{model_name} Hyperparameter Tuning Performance')
#    plt.savefig(f'{model_name}_performance.png')


# Function to display results for the best model
def display_results(model_name, X_train, X_test, y_train, y_test, y_scaler):
    best_model = joblib.load(f'{model_name}_model.pkl')
    #best_model = joblib.load(f'{model_name}_best_model.pkl')
    y_scaler = joblib.load(y_scaler)
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    y_train_pred = y_scaler.inverse_transform(y_train_pred.reshape(-1, 1))
    y_test_pred = y_scaler.inverse_transform(y_test_pred.reshape(-1, 1))
    y_train = y_scaler.inverse_transform(y_train)
    y_test = y_scaler.inverse_transform(y_test)
    
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

# Function to fit and evaluate models without hyperparameter search
def fit_and_evaluate(model_name, model_class, data):
    X_train, X_test, y_train, y_test = data

    print(f"Fitting {model_name}...")

    # Instantiate the model with default parameters
    model = model_class()

    # Fit the model
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, f'{model_name}_model.pkl')

    print(f"Model fitting complete for {model_name}.")

    # Evaluate on test set
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

# Main function for fitting models
def main():
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv')
    y_test = pd.read_csv('y_test.csv')
    y_scaler = 'y_scaler.pkl'

    models = {
        'Ridge': Ridge,
        'ElasticNet': ElasticNet,
        'BayesianRidge': BayesianRidge,
        'PassiveAggressiveRegressor': PassiveAggressiveRegressor,
        'HuberRegressor': HuberRegressor,
        'KNeighborsRegressor': KNeighborsRegressor,
        'RandomForestRegressor': RandomForestRegressor,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'ExtraTreesRegressor': ExtraTreesRegressor,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'SVR': SVR,
        'KernelRidge': KernelRidge,
        'GaussianProcessRegressor': GaussianProcessRegressor,
        'XGBRegressor': XGBRegressor,
        'LGBMRegressor': LGBMRegressor,
        'CatBoostRegressor': CatBoostRegressor,
        'AdaBoostRegressor': AdaBoostRegressor,
        'NGBRegressor': NGBRegressor
    }

    for model_name, model_class in models.items():
        print(f'Running model fitting for {model_name}')
        fit_and_evaluate(model_name, model_class, (X_train, X_test, y_train, y_test))
        #print(f'Running hyperparameter tuning for {model_name}')
        #grid_search = run_grid_search(model_name, model_class, (X_train, X_test, y_train, y_test))
        #print(f'Best configuration for {model_name}: {grid_search.best_params_}')
        #plot_performance(grid_search, model_name)
        display_results(model_name, X_train, X_test, y_train, y_test, y_scaler)

if __name__ == '__main__':
    main()
