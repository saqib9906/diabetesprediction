# Import required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the diabetes dataset
data = load_diabetes()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#shape
print("shape")
print(X_train.shape[1])
# Initialize the model
model = Sequential()

# Add layers to the model
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))  # Input layer + first hidden layer
model.add(Dense(units=32, activation='relu'))  # Second hidden layer
model.add(Dense(units=1))  # Output layer for regression (since we are predicting a continuous value)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the summary of the model
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=32)

# Evaluate the model on the test data
test_loss = model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss}')

# Predict on the test set
y_pred = model.predict(X_test)

# Print the predictions
print(y_pred)

# Plot the training loss and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
