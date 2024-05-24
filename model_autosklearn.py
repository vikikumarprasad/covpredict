import pandas as pd
import numpy as np
from autosklearn.regression import AutoSklearnRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load train and test datasets
X_train = pd.read_csv('X_train.csv', index_col=0)
X_test = pd.read_csv('X_test.csv', index_col=0)
y_train = pd.read_csv('y_train.csv', index_col=0)
y_test = pd.read_csv('y_test.csv', index_col=0)

# Ensure y is in the correct shape
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Initialize and fit the AutoSklearnRegressor
automl = AutoSklearnRegressor(time_left_for_this_task=3600,  # 1 hour
                              per_run_time_limit=300,  # 5 minutes per model
                              n_jobs=-1)  # Use all available cores

automl.fit(X_train, y_train)

# Predict on the test set
y_pred = automl.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Print the final ensemble built by auto-sklearn
print(automl.show_models())

# Optional: save the model for future use
import joblib
joblib.dump(automl, 'automl_regressor.pkl')

# Load the model (if needed later)
# automl = joblib.load('automl_regressor.pkl')
