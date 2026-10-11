
# ============================================================
# PROJECT: PREDICTIVE CHURN ANALYSIS
# Algorithm: Logistic Regression
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
)

import joblib

# ------------------------------------------------------------
# STEP 1: LOAD DATASET
# ------------------------------------------------------------

print("\nPREDICTIVE CHURN ANALYSIS")
print("=" * 50)

DATA_FILE = "customer_churn_model_v2.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# ------------------------------------------------------------
# STEP 2: PREPARE DATA
# ------------------------------------------------------------

# Check that the target column exists
if "Churn" not in df.columns:
    raise ValueError("Dataset must contain a 'Churn' column.")

# Save customer IDs for the prediction output
if "CustomerID" in df.columns:
    customer_ids = df["CustomerID"].copy()
    df = df.drop(columns=["CustomerID"])
else:
    customer_ids = pd.Series(df.index, index=df.index, name="CustomerID")

# Remove rows with missing target values
df = df.dropna(subset=["Churn"]).copy()
customer_ids = customer_ids.loc[df.index]

# Separate input features and target
X = df.drop(columns=["Churn"])
y = df["Churn"].astype(str).str.strip()

# Standardize common text labels
y = y.replace({
    "yes": "Yes",
    "no": "No",
    "YES": "Yes",
    "NO": "No",
    "1": "Yes",
    "0": "No",
    "True": "Yes",
    "False": "No"
})

if set(y.unique()) != {"Yes", "No"}:
    raise ValueError(
        "Expected Churn labels 'Yes' and 'No'. "
        f"Found: {y.unique().tolist()}"
    )

print("\nTarget distribution:")
print(y.value_counts())

# ------------------------------------------------------------
# STEP 3: IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:", numeric_features)
print("Categorical features:", categorical_features)

# ------------------------------------------------------------
# STEP 4: PREPROCESS DATA
# ------------------------------------------------------------

# Numerical data: fill missing values and scale
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

# Categorical data: fill missing values and encode
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both preprocessing pipelines
preprocessor = ColumnTransformer(transformers=[
    ("numerical", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ------------------------------------------------------------
# STEP 5: SPLIT DATA INTO 80% TRAINING AND 20% TESTING
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ------------------------------------------------------------
# STEP 6: BUILD AND TRAIN LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------

model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model trained successfully!")

# ------------------------------------------------------------
# STEP 7: PREDICT CUSTOMER CHURN
# ------------------------------------------------------------

y_pred = model.predict(X_test)

# Get probability for the positive class: Yes
classifier = model.named_steps["classifier"]
yes_index = list(classifier.classes_).index("Yes")

y_prob = model.predict_proba(X_test)[:, yes_index]

# ------------------------------------------------------------
# STEP 8: EVALUATE MODEL
# ------------------------------------------------------------

precision = precision_score(
    y_test, y_pred, pos_label="Yes", zero_division=0
)

recall = recall_score(
    y_test, y_pred, pos_label="Yes", zero_division=0
)

f1 = f1_score(
    y_test, y_pred, pos_label="Yes", zero_division=0
)

y_test_binary = (y_test == "Yes").astype(int)

roc_auc = roc_auc_score(y_test_binary, y_prob)

print("\n" + "=" * 50)
print("MODEL EVALUATION RESULTS")
print("=" * 50)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=["No", "Yes"],
    zero_division=0
))

print("Confusion Matrix (rows=actual, columns=predicted):")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["No", "Yes"]
))

# ------------------------------------------------------------
# STEP 9: PLOT AND SAVE ROC CURVE
# ------------------------------------------------------------

RocCurveDisplay.from_predictions(
    y_test_binary,
    y_prob,
    name="Logistic Regression"
)

plt.title("ROC Curve - Predictive Churn Analysis")
plt.grid(True)
plt.tight_layout()

roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=300)
plt.show()

print("ROC curve saved to:", roc_path)

# ------------------------------------------------------------
# STEP 10: EXPORT CUSTOMER CHURN PREDICTIONS
# ------------------------------------------------------------

results = pd.DataFrame({
    "CustomerID": customer_ids.loc[X_test.index].values,
    "Actual_Churn": y_test.values,
    "Predicted_Churn": y_pred,
    "Churn_Risk_Score": y_prob,
    "Churn_Risk_Percentage": (y_prob * 100).round(2)
})

results = results.sort_values(
    by="Churn_Risk_Score",
    ascending=False
)

prediction_path = os.path.join(
    OUTPUT_DIR,
    "customer_churn_predictions.csv"
)

results.to_csv(prediction_path, index=False)

print("\nTop 10 customers by predicted churn risk:")
print(results.head(10).to_string(index=False))

print("\nPredictions saved to:", prediction_path)

# ------------------------------------------------------------
# STEP 11: SAVE TRAINED MODEL
# ------------------------------------------------------------

model_path = os.path.join(
    OUTPUT_DIR,
    "customer_churn_model.pkl"
)

joblib.dump(model, model_path)

print("Trained model saved to:", model_path)

# ------------------------------------------------------------
# PROJECT COMPLETED
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 50)
