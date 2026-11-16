"""
data_cleaning.py
----------------
Turns the messy raw dataset into a clean, analysis-ready dataset.

The ten required operations are all implemented here. They are labelled with
the same numbers as in the project brief (STEP 1 ... STEP 10). The steps are
NOT executed in numerical order, because some steps need others first:

    1 Load -> 2 Head -> 3 Shape -> 4 Find missing values
    -> 6a Remove exact duplicates   (done early so copies do not distort anything)
    -> 8  Correct inconsistent formats (numbers, labels, dates must be tidy first)
    -> 6b Remove duplicate student IDs (visible only after formats are tidy)
    -> 5  Handle missing values
    -> 7  Handle outliers          (after imputing, so the quartiles are reliable)
    -> 9  Remove unnecessary columns
    -> 10 Save the cleaned CSV

Run it on its own with:   python -m src.data_cleaning
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from src import config

# Date styles found in the raw file (the first one that fits is used)
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%B %d, %Y", "%d %b %Y"]

# Lower-case spelling -> the one correct spelling
GENDER_MAP = {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}
YES_NO_MAP = {"yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes",
              "no": "No", "n": "No", "false": "No", "0": "No"}
CATEGORY_MAPS = {
    "gender": GENDER_MAP,
    "internet_access": YES_NO_MAP,
    "extra_classes": YES_NO_MAP,
    "school_type": {v.lower(): v for v in config.SCHOOL_TYPES},
    "parental_education": {v.lower(): v for v in config.EDUCATION_ORDER},
}


def _silent(message=""):
    """Default logger: prints nothing (used by the dashboard and the tests)."""
    return None


def _section(log, title):
    """Print a clear heading for a cleaning step."""
    log("")
    log("=" * 72)
    log(title)
    log("=" * 72)


# ----------------------------------------------------------------------
# Small helper functions
# ----------------------------------------------------------------------
def load_raw_data(path=config.RAW_PATH):
    """STEP 1 - Load the raw CSV file. Tokens such as '?' become missing."""
    return pd.read_csv(path, na_values=config.MISSING_TOKENS)


def missing_value_summary(df):
    """STEP 4 - Table with the number and percentage of missing values per column."""
    counts = df.isna().sum()
    summary = pd.DataFrame({
        "column": counts.index,
        "missing_count": counts.to_numpy(),
        "missing_percent": (counts.to_numpy() / len(df) * 100).round(2),
    })
    return summary.sort_values("missing_count", ascending=False).reset_index(drop=True)


def to_number(series):
    """Turn text such as '87.5%' or ' 90 ' into the number 87.5 or 90.0.
    Anything that cannot be converted becomes NaN (missing)."""
    text = series.astype("string").str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(text, errors="coerce").astype("float64")


def standardise_text(series, mapping):
    """Strip spaces, make lower-case and map to the one correct spelling.
    Unknown spellings become missing values."""
    cleaned = series.astype("string").str.strip().str.lower()
    return cleaned.map(mapping).astype(object)


def parse_date(value):
    """Try every known date style. Returns (timestamp, style_used)."""
    if pd.isna(value):
        return pd.NaT, None
    text = str(value).strip()
    for style in DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(text, style)), style
        except ValueError:
            continue
    return pd.NaT, None


def iqr_bounds(series, factor=1.5):
    """Lower and upper limits of the IQR rule: Q1 - 1.5*IQR and Q3 + 1.5*IQR."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - factor * iqr), float(q3 + factor * iqr)


# ----------------------------------------------------------------------
# The cleaning pipeline
# ----------------------------------------------------------------------
def clean_data(raw_df, log=_silent):
    """Clean the raw DataFrame. Returns (cleaned_df, report_dict).

    `log` is a function that receives text messages (use print to see them).
    """
    df = raw_df.copy()
    report = {
        "rows_before": int(len(df)),
        "cols_before": int(df.shape[1]),
    }

    # ---------------- STEP 4: identify missing values ----------------
    _section(log, "STEP 4: Identify missing values (raw data)")
    missing_before = missing_value_summary(df)
    report["missing_before"] = {r.column: int(r.missing_count) for r in missing_before.itertuples()}
    report["missing_before_total"] = int(df.isna().sum().sum())
    report["rows_with_missing_before"] = int(df.isna().any(axis=1).sum())
    log(missing_before[missing_before.missing_count > 0].to_string(index=False))
    log(f"\nTotal missing cells: {report['missing_before_total']}")

    # Facts about the formatting problems (used for the before/after table)
    report["distinct_labels_before"] = {
        col: int(raw_df[col].nunique()) for col in ["gender", "internet_access", "extra_classes", "school_type"]
    }
    report["attendance_text_values_before"] = int(
        raw_df["attendance_percent"].astype("string").str.contains("%", na=False).sum())

    # ---------------- STEP 6a: exact duplicate rows ----------------
    _section(log, "STEP 6a: Identify and remove exact duplicate records")
    report["duplicate_ids_before"] = int(raw_df["student_id"].duplicated().sum())
    exact_duplicates = int(df.duplicated().sum())
    report["exact_duplicates"] = exact_duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    log(f"Exact duplicate rows found and removed: {exact_duplicates}")
    log(f"Rows remaining: {len(df)}")

    # ---------------- STEP 8: inconsistent formats ----------------
    _section(log, "STEP 8: Correct inconsistent data formats")
    # 8a - numbers stored as text ("87.5%") -> real numbers
    for column in config.NUMERIC_COLUMNS:
        df[column] = to_number(df[column])
    log("Converted all numeric columns to numbers (removed '%' signs and spaces).")

    # 8b - labels with different spelling -> one standard spelling
    for column, mapping in CATEGORY_MAPS.items():
        before = df[column].nunique()
        df[column] = standardise_text(df[column], mapping)
        log(f"  {column}: {before} different spellings -> {df[column].nunique()} standard labels")

    # 8c - five date styles -> one real date type
    parsed = df["exam_date"].apply(parse_date)
    df["exam_date"] = pd.to_datetime([p[0] for p in parsed])
    styles_used = {p[1] for p in parsed if p[1] is not None}
    report["date_formats_before"] = len(styles_used)
    log(f"  exam_date: {len(styles_used)} different date styles -> one date format (YYYY-MM-DD)")

    # 8d - impossible values (outside the valid range) -> missing
    invalid_counts = {}
    for column, (low, high) in config.VALID_RANGES.items():
        impossible = (df[column] < low) | (df[column] > high)
        invalid_counts[column] = int(impossible.sum())
        df.loc[impossible, column] = np.nan
    report["invalid_values"] = invalid_counts
    report["invalid_total"] = int(sum(invalid_counts.values()))
    log("  Impossible values set to missing (will be filled in STEP 5):")
    for column, count in invalid_counts.items():
        if count:
            low, high = config.VALID_RANGES[column]
            log(f"    {column}: {count} value(s) outside {low}-{high}")

    df["student_id"] = df["student_id"].astype(str).str.strip().str.upper()

    # ---------------- STEP 6b: duplicate student IDs ----------------
    _section(log, "STEP 6b: Remove duplicate student IDs (re-typed records)")
    repeated_ids = int(df["student_id"].duplicated().sum())
    report["near_duplicates"] = repeated_ids
    df = df.drop_duplicates(subset="student_id", keep="first").reset_index(drop=True)
    log(f"Re-typed duplicate records removed (same student_id): {repeated_ids}")
    log(f"Rows remaining: {len(df)}")
    report["duplicates_removed_total"] = exact_duplicates + repeated_ids

    # ---------------- STEP 5: handle missing values ----------------
    _section(log, "STEP 5: Handle missing values")
    log(f"Missing cells now (after format fixes): {int(df.isna().sum().sum())}")
    imputation = {}
    for column in config.NUMERIC_COLUMNS:
        fill_value = float(df[column].median())
        n_missing = int(df[column].isna().sum())
        df[column] = df[column].fillna(fill_value)
        imputation[column] = {"method": "median", "value": round(fill_value, 2), "filled": n_missing}
    for column in config.CATEGORICAL_COLUMNS:
        fill_value = df[column].mode().iloc[0]
        n_missing = int(df[column].isna().sum())
        df[column] = df[column].fillna(fill_value)
        imputation[column] = {"method": "mode", "value": str(fill_value), "filled": n_missing}
    if df["exam_date"].isna().any():
        fill_value = df["exam_date"].mode().iloc[0]
        n_missing = int(df["exam_date"].isna().sum())
        df["exam_date"] = df["exam_date"].fillna(fill_value)
        imputation["exam_date"] = {"method": "mode", "value": str(fill_value.date()), "filled": n_missing}
    report["imputation"] = imputation
    for column, info in imputation.items():
        if info["filled"]:
            log(f"  {column}: filled {info['filled']} value(s) with the {info['method']} = {info['value']}")
    # (The columns dropped in STEP 9 are ignored here; they are not analysed.)
    kept = [c for c in df.columns if c not in config.COLUMNS_TO_DROP]
    log(f"\nMissing cells left in the analysis columns: {int(df[kept].isna().sum().sum())}")

    # ---------------- STEP 7: outliers (IQR method) ----------------
    _section(log, "STEP 7: Detect and handle outliers (IQR method, values are capped)")
    outliers = {}
    for column in config.OUTLIER_COLUMNS:
        low, high = iqr_bounds(df[column])
        below = int((df[column] < low).sum())
        above = int((df[column] > high).sum())
        # Cap at the IQR limits, but never beyond the valid range
        # (for example study hours can never be below 0).
        valid_low, valid_high = config.VALID_RANGES[column]
        df[column] = df[column].clip(lower=max(low, valid_low), upper=min(high, valid_high))
        outliers[column] = {
            "lower": round(low, 2), "upper": round(high, 2),
            "below": below, "above": above, "count": below + above,
        }
        log(f"  {column}: limits [{low:.1f}, {high:.1f}] -> {below + above} outlier(s) capped")
    report["outliers"] = outliers
    report["outliers_total"] = int(sum(o["count"] for o in outliers.values()))

    # Round the numbers neatly and make age a whole number
    for column in config.OUTLIER_COLUMNS:
        df[column] = df[column].round(1)
    df["age"] = df["age"].round().astype(int)

    # Extra columns that make the analysis easier
    df["average_score"] = df[config.SCORE_COLUMNS].mean(axis=1).round(2)
    df["grade"] = pd.cut(df["average_score"], bins=[-1, 40, 50, 60, 75, 101],
                         labels=["F", "D", "C", "B", "A"], right=False).astype(str)
    df["study_level"] = pd.cut(df["study_hours_per_week"], bins=[-np.inf, 10, 20, np.inf],
                               labels=config.STUDY_LEVEL_ORDER, right=False).astype(str)
    report["columns_added"] = ["average_score", "grade", "study_level"]

    # ---------------- STEP 9: remove unnecessary columns ----------------
    _section(log, "STEP 9: Remove unnecessary columns")
    to_drop = [c for c in config.COLUMNS_TO_DROP if c in df.columns]
    # Also drop any column that has the same value in every row
    constant = [c for c in df.columns if df[c].nunique() <= 1 and c not in to_drop]
    to_drop += constant
    df = df.drop(columns=to_drop)
    report["columns_dropped"] = to_drop
    log(f"Dropped columns: {to_drop}")

    df = df[[c for c in config.FINAL_COLUMN_ORDER if c in df.columns]]
    report["rows_after"] = int(len(df))
    report["cols_after"] = int(df.shape[1])
    report["missing_after_total"] = int(df.isna().sum().sum())
    report["missing_after"] = {c: int(v) for c, v in df.isna().sum().items()}

    # Re-check the final data against the rules (should all be zero)
    report["invalid_after"] = int(sum(
        ((df[c] < lo) | (df[c] > hi)).sum() for c, (lo, hi) in config.VALID_RANGES.items()))
    report["outliers_after"] = int(sum(
        ((df[c] < o["lower"] - 0.05) | (df[c] > o["upper"] + 0.05)).sum() for c, o in outliers.items()))
    report["distinct_labels_after"] = {c: int(df[c].nunique()) for c in report["distinct_labels_before"]}
    return df, report


# ----------------------------------------------------------------------
# Before / after comparison
# ----------------------------------------------------------------------
def build_comparison_table(report):
    """Return a small table that compares the dataset before and after cleaning."""
    rows = [
        ("Number of rows", report["rows_before"], report["rows_after"]),
        ("Number of columns", report["cols_before"], report["cols_after"]),
        ("Missing cells (total)", report["missing_before_total"], report["missing_after_total"]),
        ("Rows with at least one missing value", report["rows_with_missing_before"], 0),
        ("Exact duplicate rows", report["exact_duplicates"], 0),
        ("Repeated student IDs", report["duplicate_ids_before"], 0),
        ("Impossible values (outside valid range)", report["invalid_total"], report["invalid_after"]),
        ("Outliers detected by IQR rule", report["outliers_total"], report["outliers_after"]),
        ("Distinct spellings of 'gender'", report["distinct_labels_before"]["gender"],
         report["distinct_labels_after"]["gender"]),
        ("Distinct spellings of 'internet_access'", report["distinct_labels_before"]["internet_access"],
         report["distinct_labels_after"]["internet_access"]),
        ("Attendance values stored as text ('87%')", report["attendance_text_values_before"], 0),
        ("Date styles in 'exam_date'", report["date_formats_before"], 1),
        ("Unnecessary columns", len(report["columns_dropped"]), 0),
    ]
    return pd.DataFrame(rows, columns=["Measure", "Before cleaning", "After cleaning"])


# ----------------------------------------------------------------------
# Full run with file input / output
# ----------------------------------------------------------------------
def _to_json_safe(value):
    """Help json.dump with numpy numbers."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def run_cleaning(raw_path=config.RAW_PATH, cleaned_path=config.CLEANED_PATH,
                 report_path=config.REPORT_PATH, log=print):
    """Run all 10 steps, save the cleaned CSV and the cleaning report.
    Returns (raw_df, cleaned_df, report)."""
    _section(log, "STEP 1: Load the raw dataset")
    raw_df = load_raw_data(raw_path)
    log(f"Loaded: {raw_path}")

    _section(log, "STEP 2: First five rows of the RAW data")
    log(raw_df.head().to_string())

    _section(log, "STEP 3: Number of rows and columns (RAW data)")
    log(f"Rows: {raw_df.shape[0]}   Columns: {raw_df.shape[1]}")

    cleaned_df, report = clean_data(raw_df, log=log)

    _section(log, "STEP 10: Save the cleaned dataset as CSV")
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(cleaned_path, index=False)
    log(f"Saved: {cleaned_path}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=_to_json_safe)
    log(f"Cleaning report saved: {report_path}")

    _section(log, "DATASET AFTER CLEANING: first five rows")
    log(cleaned_df.head().to_string())
    log(f"\nRows: {cleaned_df.shape[0]}   Columns: {cleaned_df.shape[1]}")

    _section(log, "BEFORE vs AFTER CLEANING")
    log(build_comparison_table(report).to_string(index=False))
    return raw_df, cleaned_df, report


if __name__ == "__main__":
    run_cleaning()
