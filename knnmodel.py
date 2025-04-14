import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load the dataset
pima = pd.read_csv(r'C:\Users\saqib\Downloads\diabetes.csv')
X = pima.drop(['Outcome','Glucose','Pregnancies', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'], axis=1)  # Features
y = pima['Outcome']  # Target variable

print(X.head())
# Print the original class distribution of the whole dataset
print(f'Original dataset shape: {Counter(y)}')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the class distribution of the training set before resampling
print(f'Original training dataset shape: {Counter(y_train)}')

# Apply smote to the training data
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# Print the class distribution of the training set after resampling
print(f'Resampled training dataset shape: {Counter(y_train_resampled)}')

# Data preprocessing - Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNN classifier
knn = KNeighborsClassifier()

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV with StratifiedKFold
grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)

# Train the classifier with GridSearchCV on the scaled, resampled training data
grid_search.fit(X_train_scaled, y_train_resampled)

# Get the best estimator
best_knn = grid_search.best_estimator_

# Make predictions on the scaled testing data with the best estimator
y_pred_test = best_knn.predict(X_test_scaled)
# Make predictions on the scaled training data with the best estimator for accuracy check
y_pred_train = best_knn.predict(X_train_scaled)

# Calculate the accuracy of the classifier
accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_train = accuracy_score(y_train_resampled, y_pred_train)

print("Best Parameters:", grid_search.best_params_)
print("Testing Accuracy:", accuracy_test)
