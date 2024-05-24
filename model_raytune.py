import pandas as pd
import numpy as np
#import rdkit
#import GPy
#import GPflow
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem

from sklearn.model_selection import train_test_split, LeaveOneOut, KFold,  cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.externals import joblib

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

from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Setup the hyperparameter search space
regressor_configs = {
        
    'Ridge': {
        'alpha': tune.loguniform(1e-4, 1e1),
        'solver': tune.choice(['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
    },

    'ElasticNet': {
        'alpha': tune.loguniform(1e-4, 1e1),
        'l1_ratio': tune.uniform(0, 1),
        'selection': tune.choice(['cyclic', 'random'])
    },

    'BayesianRidge': {
        'alpha_1': tune.loguniform(1e-6, 1e-3),
        'alpha_2': tune.loguniform(1e-6, 1e-3),
        'lambda_1': tune.loguniform(1e-6, 1e-2),
        'lambda_2': tune.loguniform(1e-6, 1e-2)
    },

    'PassiveAggressiveRegressor': {
        'C': tune.loguniform(1e-4, 1e1),
        'loss': tune.choice(['epsilon_insensitive', 'squared_epsilon_insensitive']),
        'epsilon': tune.uniform(0, 1)
    },

    'HuberRegressor': {
        'alpha': tune.loguniform(1e-4, 1e1)
        'epsilon': tune.uniform(1.1, 2.0),
    },

    'KNeighborsRegressor': {
        'n_neighbors': tune.choice(range(1, 31)),
        'weights': tune.choice(['uniform', 'distance']),
        'algorithm': tune.choice(['auto', 'ball_tree', 'kd_tree', 'brute'])
    },

    'RandomForestRegressor': {
        'n_estimators': tune.choice([10, 50, 100, 200]),
        'max_depth': tune.choice([5, 10, 20, None]),
        'min_samples_split': tune.choice([2, 5, 10]),
        'min_samples_leaf': tune.choice([1, 2, 4])
    },

    'GradientBoostingRegressor': {
        'n_estimators': tune.choice([50, 100, 200]),
        'learning_rate': tune.uniform(0.01, 0.2),
        'max_depth': tune.choice([3, 5, 7]),
        'subsample': tune.uniform(0.5, 1.0)
    },

    'ExtraTreesRegressor': {
        'n_estimators': tune.choice([10, 50, 100, 200]),
        'max_depth': tune.choice([5, 10, 20, None]),
        'min_samples_split': tune.choice([2, 5, 10]),
        'min_samples_leaf': tune.choice([1, 2, 4])
    },

    'DecisionTreeRegressor': {
        'max_depth': tune.choice([5, 10, 20, None]),
        'min_samples_split': tune.choice([2, 5, 10]),
        'min_samples_leaf': tune.choice([1, 2, 4]),
        'max_features': tune.choice(['auto', 'sqrt', 'log2', None])
        },

    'SVR': {
        'C': tune.loguniform(1e-4, 1e2),
        'epsilon': tune.uniform(0.01, 0.2),
        'gamma': tune.choice(['auto', 'scale']),
        'degree': tune.choice([2, 3, 4]),
        'coef0': tune.uniform(0, 1),
        'max_iter': tune.choice([100, 1000, 5000, -1])
        'kernel': tune.choice(['linear', 'poly', 'rbf', 'sigmoid'])
    },

    'KernelRidge': {
        'alpha': tune.loguniform(1e-4, 1e1),
        'gamma': tune.choice(['auto', 'scale']),
        'degree': tune.choice([2, 3, 4]),
        'coef0': tune.uniform(0, 1)
        'kernel': tune.choice(['linear', 'rbf', 'laplacian', 'polynomial', 'exponential', 'chi2', 'sigmoid'])
    },

    'GaussianProcessRegressor': {
        'alpha': tune.loguniform(1e-10, 1e-1),
        'length_scale': tune.loguniform(1e-2, 1e1),
        #'n_restarts_optimizer': tune.choice([10]),
        #kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        #kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        'kernel': tune.choice(['rbf', 'matern', 'rationalquadratic'])
    },

    'XGBRegressor': {
        'n_estimators': tune.choice([50, 100, 200]),
        'learning_rate': tune.uniform(0.01, 0.2),
        'max_depth': tune.choice([3, 5, 7]),
        'subsample': tune.uniform(0.5, 1.0),
        'colsample_bytree': tune.uniform(0.5, 1.0),
        'min_child_weight': tune.choice([1, 5, 10])
    },

    'LGBMRegressor': {
        'n_estimators': tune.choice([50, 100, 200]),
        'learning_rate': tune.uniform(0.01, 0.2),
        'max_depth': tune.choice([3, 5, 7]),
        'num_leaves': tune.choice([31, 63, 127]),
        'feature_fraction': tune.uniform(0.5, 1.0),
        'bagging_fraction': tune.uniform(0.5, 1.0)
    },

    'CatBoostRegressor': {
        'iterations': tune.choice([50, 100, 200]),
        'learning_rate': tune.uniform(0.01, 0.2),
        'depth': tune.choice([4, 6, 8]),
        'l2_leaf_reg': tune.loguniform(1e-4, 1e1),
        'border_count': tune.choice([32, 64, 128])
    },

    'AdaBoostRegressor': {
        'n_estimators': tune.choice([10, 50, 100, 200]),
        'learning_rate': tune.uniform(0.01, 0.2),
        'loss': tune.choice(['linear', 'square', 'exponential'])
    },

    'NGBRegressor': {
        'n_estimators': tune.choice([50, 100, 200]),
        'learning_rate': tune.uniform(0.01, 0.2),
        'minibatch_frac': tune.uniform(0.5, 1.0),
        'natural_gradient': tune.choice([True, False])
    }

}

# Define the training function for hyperparameter tuning
def train_model(config, model_class, data, cv_type='kfold'):
    X_train, X_test, y_train, y_test = data

    # Initialize the model with given hyperparameters
    model = model_class(**config)

    # Choose the cross-validator based on cv_type
    if cv_type == 'loo':
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=5)

    # Perform cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train).reshape(-1, 1)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)

    # Report cross-validation and test performance
    tune.report(mean_cv_score=-np.mean(scores), mean_test_score=test_score)

    return model

# Wrapper function for Ray Tune
def run_tune(model_name, model_class, data, cv_type='kfold'):
    def train_model_tune(config):
        train_model(config, model_class, data, cv_type=cv_type)

    # Setup the ASHAScheduler search with early stopping
    scheduler = ASHAScheduler(
        max_t=100,
        grace_period=10,
        reduction_factor=2
    )

    analysis = tune.run(
        train_model_tune,
        config=regressor_configs[model_name],
        num_samples=10,
        local_dir='ray_results',
        resources_per_trial={'cpu': 1},
        metric='mean_cv_score',
        mode='min',
        scheduler=scheduler
    )

    best_config = analysis.best_config
    best_model = train_model(best_config, model_class, data)
    joblib.dump(best_model, f'{model_name}_best_model.pkl')

    return analysis

# Function to generate and save performance plots
def plot_performance(analysis, model_name):
    df = analysis.results_df
    df[['mean_cv_score', 'mean_test_score']].plot(title=f'{model_name} Hyperparameter Tuning Performance')
    plt.savefig(f'{model_name}_performance.png')

# Main function for data preparation and running hyperparameter search
def main():
    # Load your dataset file (replace with the actual data file)
    data = pd.read_csv('your_data_file.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    # Split data into training and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ray.init(ignore_reinit_error=True)

    # Models to tune with updated regressors
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
        analysis = run_tune(model_name, model_class, (X_train, X_test, y_train, y_test))
        print(f'Best configuration for {model_name}: {analysis.best_config}')
        plot_performance(analysis, model_name)

    ray.shutdown()

if __name__ == '__main__':
    main()
