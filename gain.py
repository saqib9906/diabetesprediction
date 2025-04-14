import pandas as pd
import numpy as np

# Assuming the dataset is available in the environment
df = pd.read_csv(r'C:\Users\saqib\Downloads\diabetes.csv')


# Calculate entropy of the target variable
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = -np.sum([(counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy

# Calculate information gain for each feature
def info_gain(data, split_attribute_name, target_name="Outcome"):
    # Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])

    # Calculate the values and the corresponding counts for the split attribute
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    # Calculate the weighted entropy
    weighted_entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])

    # Calculate the information gain
    information_gain = total_entropy - weighted_entropy
    return information_gain

# Calculate entropy of the target variable
target_entropy = entropy(df["Outcome"])

# Calculate information gain for each feature
features = df.columns[:-1]
information_gains = {feature: info_gain(df, feature) for feature in features}

(target_entropy, information_gains)
# Display the information gains
print("Information Gain for each feature:")
for feature, gain in information_gains.items():
    print(f"{feature}: {gain}")