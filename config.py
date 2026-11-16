"""
config.py
---------
Central place for every path, column list and setting used in the project.

Keeping these in ONE file means the other modules stay short and you only
have to change a value in one place (for example a valid range).
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Folder and file locations
# ----------------------------------------------------------------------
# __file__ is src/config.py, so .parent.parent is the project root folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
GRAPH_DIR = OUTPUT_DIR / "graphs"

RAW_PATH = DATASET_DIR / "raw_data.csv"
CLEANED_PATH = DATASET_DIR / "cleaned_data.csv"
REPORT_PATH = OUTPUT_DIR / "cleaning_report.json"

# ----------------------------------------------------------------------
# Text that should be treated as "missing" when the CSV is loaded.
# (Pandas already treats empty cells and "N/A" as missing.)
# ----------------------------------------------------------------------
MISSING_TOKENS = ["?", "-", "--", "None", "null"]

# ----------------------------------------------------------------------
# Column groups
# ----------------------------------------------------------------------
NUMERIC_COLUMNS = [
    "age",
    "study_hours_per_week",
    "attendance_percent",
    "previous_score",
    "math_score",
    "science_score",
    "english_score",
]
CATEGORICAL_COLUMNS = [
    "gender",
    "school_type",
    "parental_education",
    "internet_access",
    "extra_classes",
]
SCORE_COLUMNS = ["math_score", "science_score", "english_score"]

# Numeric columns used for statistics and correlation (after cleaning)
ANALYSIS_COLUMNS = NUMERIC_COLUMNS + ["average_score"]

# ----------------------------------------------------------------------
# Cleaning rules
# ----------------------------------------------------------------------
# Rule 1 - "impossible value" limits (domain knowledge).
# Anything outside these limits cannot be real, so it is treated as missing.
VALID_RANGES = {
    "age": (10, 30),
    "study_hours_per_week": (0, 168),   # a week only has 168 hours
    "attendance_percent": (0, 100),
    "previous_score": (0, 100),
    "math_score": (0, 100),
    "science_score": (0, 100),
    "english_score": (0, 100),
}

# Rule 2 - statistical outliers (IQR method) are checked in these columns.
OUTLIER_COLUMNS = [
    "study_hours_per_week",
    "attendance_percent",
    "previous_score",
    "math_score",
    "science_score",
    "english_score",
]

# Columns that add no value to the analysis (identity / admin / constant).
COLUMNS_TO_DROP = [
    "student_name",         # personal identifier - not needed for analysis
    "data_entry_operator",  # admin detail
    "internal_remarks",     # free-text notes, mostly empty
    "record_source",        # same value in every row (constant)
]

# ----------------------------------------------------------------------
# Allowed category values (used for standardising text)
# ----------------------------------------------------------------------
SCHOOL_TYPES = ["Government", "Aided", "Private"]
EDUCATION_ORDER = ["High School", "Diploma", "Bachelor's", "Master's"]
GRADE_ORDER = ["A", "B", "C", "D", "F"]
STUDY_LEVEL_ORDER = ["Low", "Medium", "High"]

# Column order of the final cleaned dataset
FINAL_COLUMN_ORDER = [
    "student_id", "gender", "age", "school_type", "parental_education",
    "internet_access", "extra_classes", "study_hours_per_week",
    "attendance_percent", "previous_score", "math_score", "science_score",
    "english_score", "exam_date", "average_score", "grade", "study_level",
]

# ----------------------------------------------------------------------
# Friendly names for chart labels
# ----------------------------------------------------------------------
LABELS = {
    "age": "Age (years)",
    "study_hours_per_week": "Study hours per week",
    "attendance_percent": "Attendance (%)",
    "previous_score": "Previous exam score",
    "math_score": "Math score",
    "science_score": "Science score",
    "english_score": "English score",
    "average_score": "Average score",
    "gender": "Gender",
    "school_type": "School type",
    "parental_education": "Parental education",
    "internet_access": "Internet access",
    "extra_classes": "Extra classes",
    "grade": "Grade",
    "study_level": "Study level",
}

# ----------------------------------------------------------------------
# Data dictionary: (column, data type, description)
# ----------------------------------------------------------------------
COLUMN_INFO = [
    ("student_id", "Text", "Unique ID of the student (STU1001 ...)"),
    ("student_name", "Text", "Synthetic student name (personal identifier)"),
    ("gender", "Category", "Male or Female"),
    ("age", "Number", "Age of the student in years"),
    ("school_type", "Category", "Government, Aided or Private school"),
    ("parental_education", "Category", "Highest education of a parent"),
    ("internet_access", "Category", "Does the student have internet at home? (Yes/No)"),
    ("extra_classes", "Category", "Does the student attend extra tuition classes? (Yes/No)"),
    ("study_hours_per_week", "Number", "Self-study hours per week"),
    ("attendance_percent", "Number", "Class attendance in percent"),
    ("previous_score", "Number", "Percentage scored in the previous exam"),
    ("math_score", "Number", "Math marks out of 100"),
    ("science_score", "Number", "Science marks out of 100"),
    ("english_score", "Number", "English marks out of 100"),
    ("exam_date", "Date", "Date on which the exam was written"),
    ("data_entry_operator", "Text", "Who typed the record (admin detail)"),
    ("internal_remarks", "Text", "Free-text note added during entry"),
    ("record_source", "Text", "Source form name (same in every row)"),
    ("average_score", "Number (derived)", "Mean of math, science and english scores"),
    ("grade", "Category (derived)", "A (75+), B (60-74), C (50-59), D (40-49), F (<40)"),
    ("study_level", "Category (derived)", "Low (<10 h), Medium (10-19 h), High (20+ h) per week"),
]
