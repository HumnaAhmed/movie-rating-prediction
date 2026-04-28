# =========================
# Movie Rating Prediction
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================
# 1. Load Dataset
# ============================================

df = pd.read_csv("IMDB-Movie-Data.csv")

print("Columns:", df.columns)
print(df.head())

# ============================================
# 2. Preprocessing
# ============================================

# Select useful features
df = df[['Runtime (Minutes)', 'Votes', 'Revenue (Millions)', 'Rating']]

# Remove missing values
df = df.dropna()

# Features & target
X = df[['Runtime (Minutes)', 'Votes', 'Revenue (Millions)']]
y = df['Rating']

# ============================================
# 3. Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 4. Model Training
# ============================================

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# ============================================
# 5. Predictions
# ============================================

y_pred = model.predict(X_test)

# ============================================
# 6. Evaluation
# ============================================

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# ============================================
# 7. Visualization
# ============================================

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings")
plt.grid()
plt.show()