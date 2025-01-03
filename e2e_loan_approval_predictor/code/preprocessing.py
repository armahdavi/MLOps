# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 20:00:22 2025

@author: alima
"""

import os
import zipfile
import pandas as pd
import json

# Step 1: Data retriavela from Kaggle
project_root = r'C:\Users\alima\code\e2e_loan_approval_predictor'
data_dir = os.path.join(project_root, 'data', 'raw')
os.makedirs(data_dir, exist_ok = True)

## Ensure kaggle.json is set up correctly
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')

## Command to download the dataset
os.system(f'kaggle datasets download -d architsharma01/loan-approval-prediction-dataset -p {data_dir}')

## Unzip the dataset in the data directory
zip_path = os.path.join(data_dir, 'loan-approval-prediction-dataset.zip')
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

## Load the CSV into a DataFrame
csv_path = os.path.join(data_dir, 'loan_approval_dataset.csv')


# Step 2: Clean and preprocess rewa data (ETL)
df = pd.read_csv(csv_path)
df.columns = [col.strip() for col in df.columns]
categorical_columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()
df[categorical_columns] = df[categorical_columns].apply(lambda col: col.str.strip())


## Create a dictionary with unique categories for each categorical column
encoded_json = {
    col: {value: idx for idx, value in enumerate(df[col].unique())}
    for col in categorical_columns
}

## Convert the dictionary to JSON
categories_json = json.dumps(encoded_json, indent = 4)

    
processed_folder = os.path.join(project_root, 'data', 'processed')
os.makedirs(processed_folder, exist_ok=True)  # Create the folder if it doesn't exist
json_file_path = os.path.join(processed_folder, 'encoded_json.json')

# Step 3: Save df and the JSON to the processed folder
with open(json_file_path, 'w') as file:
    file.write(categories_json)

for col, mapping in encoded_json.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

df.to_csv(os.path.join(processed_folder, 'processed_ml.csv'), index = False)

if __name__ == "__main__":
    print("This script is running directly")