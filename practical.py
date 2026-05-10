# =========================================
# PRACTICAL: CLASSIFICATION USING ML
# Algorithm: Logistic Regression
# =========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================================
# Q1: DATA ACQUISITION & UNDERSTANDING
# =========================================

# Load Dataset using pandas
df = pd.read_csv("data.csv")      # CHANGE HERE

# Display First 5 Rows
print("First 5 Rows:")
print(df.head())

# Display Last 5 Rows
print("\nLast 5 Rows:")
print(df.tail())

# Display Column Names
print("\nColumn Names:")
print(df.columns)

#convert taget column if it is not numerical
df["target_column"] = df["target_column"].map({"Yes":1, "No":0})  #change here(also change val yes,no)

# Observations
print("\nObservations:")
print("1. Dataset contains multiple features.")
print("2. Dataset contains target/output column.")

# =========================================
# Q2: DATA VISUALIZATION
# =========================================

# Scatter Plot
plt.scatter(df['column1'], df['column2'])     # CHANGE HERE

plt.title("Scatter Plot")
plt.xlabel("column1")                         # CHANGE HERE
plt.ylabel("column2")                         # CHANGE HERE
plt.show()

# Histogram
plt.hist(df['column3'])                       # CHANGE HERE

plt.title("Histogram")
plt.xlabel("column3")                         # CHANGE HERE
plt.ylabel("Frequency")
plt.show()

# Graph Observations
print("\nGraph Observations:")
print("1. Scatter plot shows relationship between two variables.")
print("2. Histogram shows data distribution.")

# =========================================
# Q3: FEATURE PREPARATION
# =========================================

# Input Features (drop columns which are not numerical)
X = df.drop('target_column', axis=1)          # CHANGE HERE
#X = df.drop(["target_column", "column"], axis=1)     # CHANGE HERE

# Output Variable
y = df['target_column']                       # CHANGE HERE

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain-Test Split Completed")

# Why splitting is required
print("\nWhy Train-Test Split?")
print("Training data is used to train the model.")
print("Testing data is used to check model accuracy.")

# =========================================
# Q4: CLASSIFICATION
# =========================================

# Create Logistic Regression Model
model = LogisticRegression()

# Train Model
model.fit(X_train, y_train)

# Predict Output
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nPredicted Values:")
print(y_pred)

print("\nAccuracy:")
print(accuracy * 100)
