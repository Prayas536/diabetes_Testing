import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("diabetes_dataset.csv")

# Target (what we want to predict)
y = df["diabetes"]

# Inputs (everything except target)
X = df.drop(columns=["diabetes"])

# Separate numeric and text columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "bool"]).columns

# Preprocessing: fill missing + encode text
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# Model: easiest and good baseline
# model = LogisticRegression(max_iter=1000) // for more balanced datasets~
model = LogisticRegression(max_iter=1000, class_weight="balanced")


# Full pipeline (preprocess + model)
clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))
