import pandas as pd
import numpy as np

# Load your dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, names=column_names)

# Replace zeros in specific columns with NaN
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)

# Function to calculate range and number of outliers
def calculate_range_and_outliers(column):
    col_range = (df[column].min(), df[column].max())
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    outliers = (np.abs(z_scores) > 3).sum()
    return col_range, outliers

# Compute for each attribute
attributes = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
preprocessing_details = []
for attr in attributes:
    col_range, outliers = calculate_range_and_outliers(attr)
    preprocessing_details.append((attr, col_range, outliers))

# Print results
for attr, col_range, outliers in preprocessing_details:
    print(f'{attr}: Range: {col_range}, Number of Outliers: {outliers}')

