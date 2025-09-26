🏠 Housing Price Prediction Using Linear Regression

Predict housing prices using machine learning! This project covers data preprocessing, exploratory data analysis (EDA), Linear Regression model building, and evaluation with visualization.


---

📂 Dataset

The dataset (Housing.csv) contains features of houses and their target price.

Make sure the CSV file path is correct before running the code.

Target column is assumed to be price.



---

🛠️ Project Steps

1️⃣ Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


---

2️⃣ Load and Preview Dataset

file_path = r"C:\Users\Dimple.S\Downloads\Housing.csv"
df = pd.read_csv(file_path)

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())


---

3️⃣ Data Preprocessing

Convert categorical features to numeric

Fill missing values

Scale features for uniformity


# Handle categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
label_enc = LabelEncoder()
for col in categorical_cols:
    df[col] = label_enc.fit_transform(df[col])

# Fill missing values
df.fillna(df.mean(), inplace=True)

# Feature scaling
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


---

4️⃣ Exploratory Data Analysis (EDA)

Correlation Heatmap

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

Helps identify relationships between features.

Feature Distributions

df.hist(figsize=(12,10), bins=30)
plt.suptitle("Feature Distributions")
plt.show()

Understand the spread and distribution of data.


---

5️⃣ Build Linear Regression Model

target_col = 'price' if 'price' in df.columns else df.columns[-1]
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


---

6️⃣ Model Evaluation

print("\nModel Performance:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

MSE: Measures average squared difference between predicted and actual prices.

R² Score: Measures how well the model explains the variance in prices.



---

7️⃣ Actual vs Predicted Prices

plt.figure(figsize=(8,5))
plt.scatter
