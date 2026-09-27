import pandas as pd
import numpy as np

# ============================================================
# DATA CLEANING & STRUCTURAL VALIDATION
# ============================================================

print("=" * 60)
print("DATA CLEANING & STRUCTURAL VALIDATION")
print("=" * 60)


# ============================================================
# STEP 1: LOAD RAW DATASET
# ============================================================

df = pd.read_csv("Raw_clean_dataset.csv")

print("\nSTEP 1: RAW DATASET LOADED")
print("-" * 60)

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


# ============================================================
# STEP 2: DISPLAY FIRST FEW RECORDS
# ============================================================

print("\nSTEP 2: FIRST 5 RECORDS")
print("-" * 60)

print(df.head())


# ============================================================
# STEP 3: INSPECT DATASET
# ============================================================

print("\nSTEP 3: DATASET INFORMATION")
print("-" * 60)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()


# ============================================================
# STEP 4: CHECK MISSING VALUES
# ============================================================

print("\nSTEP 4: MISSING VALUE CHECK")
print("-" * 60)

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal Missing Values:", missing_values.sum())


# ============================================================
# STEP 5: CHECK DUPLICATE RECORDS
# ============================================================

print("\nSTEP 5: DUPLICATE RECORD CHECK")
print("-" * 60)

duplicate_count = df.duplicated().sum()

print("Number of duplicate records:", duplicate_count)

if duplicate_count > 0:
    print("\nDuplicate records:")
    print(df[df.duplicated(keep=False)])
else:
    print("No duplicate records found.")


# ============================================================
# STEP 6: CHECK UNIQUE VALUES IN CATEGORICAL COLUMNS
# ============================================================

print("\nSTEP 6: CATEGORICAL VALUE CHECK")
print("-" * 60)

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    print("\n", column)
    print(df[column].unique())


# ============================================================
# STEP 7: STANDARDIZE COLUMN NAMES
# ============================================================

print("\nSTEP 7: STANDARDIZING COLUMN NAMES")
print("-" * 60)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("Standardized column names:")
print(df.columns.tolist())


# ============================================================
# STEP 8: CLEAN TEXT VALUES
# ============================================================

print("\nSTEP 8: CLEANING TEXT VALUES")
print("-" * 60)

text_columns = df.select_dtypes(include="object").columns

for column in text_columns:
    df[column] = df[column].str.strip()

print("Leading and trailing spaces removed.")


# ============================================================
# STEP 9: STANDARDIZE CATEGORICAL VALUES
# ============================================================

print("\nSTEP 9: STANDARDIZING CATEGORICAL VALUES")
print("-" * 60)

categorical_columns = [
    "region",
    "subscriptiontype",
    "contracttype",
    "paymentmethod",
    "autopay",
    "churn"
]

for column in categorical_columns:
    if column in df.columns:
        df[column] = df[column].str.title()

print("Categorical values standardized.")


# ============================================================
# STEP 10: HANDLE UNKNOWN VALUES
# ============================================================

print("\nSTEP 10: CHECKING UNKNOWN VALUES")
print("-" * 60)

for column in df.select_dtypes(include="object").columns:

    unknown_count = (
        df[column]
        .astype(str)
        .str.lower()
        .eq("unknown")
        .sum()
    )

    if unknown_count > 0:
        print(
            column,
            "contains",
            unknown_count,
            "unknown value(s)"
        )


# ============================================================
# STEP 11: CONVERT NUMERICAL COLUMNS
# ============================================================

print("\nSTEP 11: VALIDATING NUMERICAL DATA TYPES")
print("-" * 60)

numeric_columns = [
    "age",
    "tenuremonths",
    "monthlychargesinr",
    "totalchargesinr",
    "supporttickets",
    "latepayments"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

print("\nData types after conversion:")
print(df.dtypes)


# ============================================================
# STEP 12: CHECK NEW MISSING VALUES
# ============================================================

print("\nSTEP 12: CHECKING MISSING VALUES AFTER TYPE CONVERSION")
print("-" * 60)

print(df.isnull().sum())


# ============================================================
# STEP 13: HANDLE MISSING VALUES
# ============================================================

print("\nSTEP 13: HANDLING MISSING VALUES")
print("-" * 60)

# Region
if "region" in df.columns:
    df["region"] = df["region"].fillna("Unknown")

# Numerical columns
numerical_columns = [
    "age",
    "tenuremonths",
    "monthlychargesinr",
    "totalchargesinr",
    "supporttickets",
    "latepayments"
]

for column in numerical_columns:

    if column in df.columns:

        if df[column].isnull().sum() > 0:

            median_value = df[column].median()

            df[column] = df[column].fillna(
                median_value
            )

            print(
                column,
                "missing values replaced with median:",
                median_value
            )

print("\nMissing values after handling:")
print(df.isnull().sum())


# ============================================================
# STEP 14: CHECK INVALID VALUES
# ============================================================

print("\nSTEP 14: INVALID VALUE CHECK")
print("-" * 60)

# Age
if "age" in df.columns:

    invalid_age = df[
        (df["age"] < 0) |
        (df["age"] > 100)
    ]

    print(
        "Invalid age records:",
        len(invalid_age)
    )


# Tenure
if "tenuremonths" in df.columns:

    invalid_tenure = df[
        df["tenuremonths"] < 0
    ]

    print(
        "Invalid tenure records:",
        len(invalid_tenure)
    )


# Monthly charges
if "monthlychargesinr" in df.columns:

    invalid_monthly = df[
        df["monthlychargesinr"] < 0
    ]

    print(
        "Invalid monthly charges:",
        len(invalid_monthly)
    )


# Total charges
if "totalchargesinr" in df.columns:

    invalid_total = df[
        df["totalchargesinr"] < 0
    ]

    print(
        "Invalid total charges:",
        len(invalid_total)
    )


# Support tickets
if "supporttickets" in df.columns:

    invalid_tickets = df[
        df["supporttickets"] < 0
    ]

    print(
        "Invalid support ticket records:",
        len(invalid_tickets)
    )


# Late payments
if "latepayments" in df.columns:

    invalid_late = df[
        df["latepayments"] < 0
    ]

    print(
        "Invalid late payment records:",
        len(invalid_late)
    )


# ============================================================
# STEP 15: OUTLIER DETECTION USING IQR
# ============================================================

print("\nSTEP 15: OUTLIER DETECTION")
print("-" * 60)

outlier_columns = [
    "age",
    "tenuremonths",
    "monthlychargesinr",
    "totalchargesinr",
    "supporttickets",
    "latepayments"
]

for column in outlier_columns:

    if column in df.columns:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[
            (df[column] < lower_bound) |
            (df[column] > upper_bound)
        ]

        print(
            column,
            "-> Possible outliers:",
            len(outliers)
        )


# ============================================================
# STEP 16: REMOVE DUPLICATES
# ============================================================

print("\nSTEP 16: REMOVING DUPLICATES")
print("-" * 60)

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

removed_duplicates = (
    before_duplicates -
    after_duplicates
)

print(
    "Duplicate records removed:",
    removed_duplicates
)


# ============================================================
# STEP 17: FINAL VALIDATION
# ============================================================

print("\nSTEP 17: FINAL VALIDATION")
print("-" * 60)

print("Final number of rows:", df.shape[0])
print("Final number of columns:", df.shape[1])

print("\nFinal missing values:")
print(df.isnull().sum())

print(
    "\nFinal duplicate count:",
    df.duplicated().sum()
)

print("\nFinal data types:")
print(df.dtypes)


# ============================================================
# STEP 18: DISPLAY CLEAN DATA
# ============================================================

print("\nSTEP 18: CLEANED DATASET")
print("-" * 60)

print(df.head())


# ============================================================
# STEP 19: RECORD CLEANING DECISIONS
# ============================================================

print("\nSTEP 19: CLEANING DECISIONS")
print("-" * 60)

print("1. Loaded the raw customer churn CSV dataset.")
print("2. Inspected rows, columns and data types.")
print("3. Checked all columns for missing values.")
print("4. Checked for duplicate records.")
print("5. Standardized column names.")
print("6. Removed unnecessary spaces from text values.")
print("7. Standardized categorical text values.")
print("8. Checked unknown values.")
print("9. Converted numerical columns to numeric data types.")
print("10. Handled missing numerical values using the median.")
print("11. Checked age, tenure, charges and ticket values for invalid values.")
print("12. Detected possible numerical outliers using the IQR method.")
print("13. Removed duplicate records if present.")
print("14. Performed final validation.")
print("15. Exported the cleaned dataset.")


# ============================================================
# STEP 20: EXPORT CLEANED DATASET
# ============================================================

print("\nSTEP 20: EXPORTING CLEAN DATASET")
print("-" * 60)

output_file = "cleaned_customer_churn.csv"

df.to_csv(
    output_file,
    index=False
)

print(
    "Cleaned dataset saved successfully as:",
    output_file
)


# ============================================================
# END
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("=" * 60)