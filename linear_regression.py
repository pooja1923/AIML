# =========================================
# Use if all columns are numeric
# PRACTICAL: REGRESSION USING ML
# Algorithm: Linear Regression
# =========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================================
# Q1: DATA ACQUISITION & UNDERSTANDING
# =========================================

# Load Dataset
df = pd.read_csv("data.csv")  #change here

# Display First 5 Rows
print("First 5 Rows:")
print(df.head())

# Display Last 5 Rows
print("\nLast 5 Rows:")
print(df.tail())

# Display Column Names
print("\nColumn Names:")
print(df.columns)

# Observations
print("\nObservations:")
print("1. Dataset contains numerical data.")
print("2. Target variable is continuous.")
print("3. Regression is used for prediction.")

# =========================================
# Q2: DATA VISUALIZATION
# =========================================

# Scatter Plot
plt.scatter(df['column1'], df['target_column'])  #chnage here

plt.title("Scatter Plot")
plt.xlabel("column1")       #change here
plt.ylabel("target_column") #change here
plt.show()

# Histogram
plt.hist(df['target_column']) #change here

plt.title("Histogram")
plt.xlabel("target_column")   #change here
plt.ylabel("Frequency")
plt.show()

# Graph Observations
print("\nGraph Observations:")
print("1. Scatter plot shows relationship between feature and target.")
print("2. Histogram shows distribution of feature column.")

# =========================================
# Q3: FEATURE PREPARATION
# =========================================

# Input Features
X = df.drop(['target_column'], axis=1)  #change here

# Output Variable
y = df['target_column']   #change here

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain-Test Split Completed")

# Why splitting is required
print("\nWhy Train-Test Split?")
print("Training data is used to train the model.")
print("Testing data is used to evaluate the model.")

# =========================================
# Q4: REGRESSION
# =========================================

# Create Linear Regression Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predict Output
y_pred = model.predict(X_test)

# =========================================
# Q5: MODEL EVALUATION
# =========================================

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# R2 Score
r2 = r2_score(y_test, y_pred)

print("\nPredicted Values:")
print(y_pred)

print("\nMean Squared Error:")
print(mse)

print("\nR2 Score:")
print(r2)

rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("\nRoot Mean Squared Error (RMSE):")
print(rmse)
