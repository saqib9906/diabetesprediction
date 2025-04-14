import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt
# Load the dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Separate majority and minority classes
df_majority = df[df['Outcome'] == 0]
df_minority = df[df['Outcome'] == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=42)  # reproducible results

# Combine majority class with upsampled minority class
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
print("\nBalanced class distribution (Oversampling):")
print(df_balanced['Outcome'].value_counts())

# Plot the balanced class distribution
df_balanced['Outcome'].value_counts().plot(kind='bar')
plt.title('Balanced Class Distribution (Oversampling)')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
