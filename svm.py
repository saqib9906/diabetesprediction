# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#r'C:\Users\saqib\Downloads\diabetes.csv'
# Load the dataset
pima_data = pd.read_csv(r'C:\Users\saqib\Downloads\diabetes.csv')

# Separate features (X) and target variable (y)
X = pima_data.drop('Outcome', axis=1)  # Features
y = pima_data['Outcome']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the SVM classifier
svm_classifier = SVC(random_state=42)

# Train the classifier on the training data
svm_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = svm_classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
