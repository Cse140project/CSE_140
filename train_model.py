import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import os

# Load CSV
df = pd.read_csv("student_data.csv")

# Features and target
X = df[["school", "area", "gender", "caste", "standard"]]
y = df["dropout"]

# All are categorical
categorical_features = ["school", "area", "gender", "caste", "standard"]

# Preprocessing (OneHotEncoding for categories)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit model
model.fit(X_train, y_train)

# Save model inside model folder
os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✔ Model trained and saved as model/model.pkl")

