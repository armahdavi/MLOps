# -*- coding: utf-8 -*-
"""
Logistics regression to predict loan approval
Accuracy : ~80%

@author: alima
"""

import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import warnings

# Turn off warnings
warnings.filterwarnings('ignore')

# Step 1: ETL
project_root = r'C:\Users\alima\code\e2e_loan_approval_predictor'
file_path = os.path.join(project_root, 'data', 'processed\processed_ml.csv')
df = pd.read_csv(file_path)
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Step 2: 5fold cv over training set (training and testing together)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

## Develop logistics regression model 
log_reg = LogisticRegression(max_iter = 1000)
param_grid = {
    'penalty': ['l1', 'l2'],  # Types of regularization
    'C': [0.01, 0.1, 1, 10, 100],                  # Inverse regularization strength
}

grid_search = GridSearchCV(
    estimator = log_reg,
    param_grid = param_grid,
    scoring = 'accuracy',       # Use accuracy as the evaluation metric
    cv = 5,                     # 5-fold cross-validation
    verbose = 1,                # Print progress
    n_jobs = -1                 # Use all available CPU cores
)
grid_search.fit(X_train, y_train)

## Extract best results
results = grid_search.cv_results_

best_index = grid_search.best_index_  # Index of the best parameter combination
best_mean = results['mean_test_score'][best_index]
best_std = results['std_test_score'][best_index]
best_params = grid_search.best_params_

print(f"Best Parameters: {best_params}")
print(f"Mean Score: {best_mean:.4f}, Std Dev: {best_std:.4f}")


# Step 3: Evaluate the test set (basically serv)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", test_accuracy)


# Step 4: Save the model
model_path = os.path.join(os.path.join(project_root, 'data', 'processed', 'logistics_regression.pkl'))
with open(model_path, 'wb') as file:
    pickle.dump(best_model, file)
print(f"Model saved at: {model_path}")

if __name__ == "__main__":
    print("This script is running directly")
