# =========================================
# PRACTICAL: CLASSIFICATION USING ML
# Dataset: Iris Dataset
# Algorithm: Logistic Regression
# =========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================================
# Q1: DATA ACQUISITION & UNDERSTANDING
# =========================================

# Load Iris Dataset
iris = load_iris()

# Create DataFrame using pandas
df = pd.DataFrame(iris.data, columns=iris.feature_names) 

# Add target column
df['target'] = iris.target 

# Add flower names 
df['flower_name'] = df['target'].map({
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
})

# Display First 5 Rows
print("First 5 Rows:")
print(df.head())

# Display Last 5 Rows
print("\nLast 5 Rows:")
print(df.tail())

# Observations
print("\nObservations:")
print("1. Dataset contains flower measurements.") 
print("2. There are 3 flower classes: Setosa, Versicolor, Virginica.") 

# =========================================
# Q2: DATA VISUALIZATION
# =========================================

# Scatter Plot
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'])

plt.title("Scatter Plot")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()

# Histogram
plt.hist(df['sepal width (cm)'])

plt.title("Histogram")
plt.xlabel("Sepal Width")
plt.ylabel("Frequency")
plt.show()

# Graph Observations
print("\nGraph Observations:")
print("1. Scatter plot shows relationship between sepal and petal length.")
print("2. Histogram shows distribution of sepal width.")

# =========================================
# Q3: FEATURE PREPARATION
# =========================================

# Input Features
X = df.drop(['target', 'flower_name'], axis=1)

# Output Variable
y = df['target']

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
