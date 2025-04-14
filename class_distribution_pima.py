import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Count the occurrences of each class in the target variable
class_distribution = df['Outcome'].value_counts()

# Display the class distribution
print("\nClass distribution:")
print(class_distribution)

# Calculate the percentage of each class
class_percentage = df['Outcome'].value_counts(normalize=True) * 100

# Display the class percentages
print("\nClass distribution percentage:")
print(class_percentage)

# Plot the class distribution
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

# Plot the class distribution percentage
plt.figure(figsize=(8, 6))
class_percentage.plot(kind='bar')
plt.title('Class Distribution Percentage')
plt.xlabel('Class')
plt.ylabel('Percentage')
plt.show()
