import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
pima_data = pd.read_csv(r'C:\Users\saqib\Downloads\diabetes.csv')

# Separate features (X) and target variable (y)
X = pima_data.drop('Outcome', axis=1)  # Features
y = pima_data['Outcome']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize RandomOverSampler
ROS = RandomOverSampler(random_state=0)

# Resample the training data
X_train_ROS, y_train_ROS = ROS.fit_resample(X_train, y_train)

# Check the shapes of resampled data
print("Shape of X_train after resampling:", X_train_ROS.shape)
print("Shape of y_train after resampling:", y_train_ROS.shape)

# Train a logistic regression classifier on the resampled data
logistic_regression_ros = LogisticRegression(max_iter=1000, random_state=42)
logistic_regression_ros.fit(X_train_ROS, y_train_ROS)

# Make predictions on the testing data
y_pred_test_ros = logistic_regression_ros.predict(X_test)

# Calculate the accuracy of the classifier on the test set
accuracy_ros = accuracy_score(y_test, y_pred_test_ros)
print("Accuracy with RandomOverSampler:", accuracy_ros)

# Print a classification report for more detailed performance metrics
print("Classification Report with RandomOverSampler:")
print(classification_report(y_test, y_pred_test_ros))
