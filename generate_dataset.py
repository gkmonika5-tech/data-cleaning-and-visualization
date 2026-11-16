"""
generate_dataset.py
-------------------
Creates the RAW student-performance dataset used in this project.

IMPORTANT: THE DATA IS SYNTHETIC (computer generated). No real student is
described. Names are random combinations of common first and last names.

Why synthetic?  A real dataset would not necessarily contain every data-quality
problem this project wants to demonstrate. Here we first create a tidy dataset
and then deliberately "damage" it in realistic ways:

    * missing values            (about 5% of the cells in 10 columns)
    * impossible values         (age 150, marks of 999, attendance of 250%)
    * statistical outliers      (study hours of 120 per week, very low marks)
    * inconsistent formats      (Male / male / M, "85%" vs 85, mixed date styles)
    * duplicate records         (20 exact copies + 6 re-typed copies)
    * unnecessary columns       (name, operator, remarks, constant source column)

A fixed random seed is used, so the same file is produced every time.

Run it on its own with:   python -m src.generate_dataset
"""

import numpy as np
import pandas as pd

from src import config

SEED = 42
N_STUDENTS = 500

FIRST_NAMES = [
    "Arun", "Priya", "Karthik", "Divya", "Suresh", "Meena", "Vignesh", "Lakshmi",
    "Rahul", "Anjali", "Naveen", "Keerthana", "Manoj", "Swathi", "Harish",
    "Nandhini", "Ajay", "Pooja", "Dinesh", "Sneha", "Gokul", "Revathi",
    "Praveen", "Janani", "Sanjay", "Deepika", "Vijay", "Kavya", "Rohit", "Ishwarya",
]
LAST_NAMES = [
    "Kumar", "Raj", "Subramanian", "Iyer", "Nair", "Reddy", "Sharma", "Pillai",
    "Murugan", "Krishnan", "Selvam", "Natarajan", "Venkatesh", "Balaji", "Menon",
    "Gupta", "Rao", "Chandran", "Mohan", "Bose",
]

# Different ways the same label may be typed by different people
GENDER_VARIANTS = {"Male": ["male", "M", "MALE", "m"], "Female": ["female", "F", "FEMALE", "f"]}
YES_NO_VARIANTS = {"Yes": ["yes", "Y", "YES", "y"], "No": ["no", "N", "NO", "n"]}

# Different date styles that were typed into the form
DATE_STYLES = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%B %d, %Y", "%d %b %Y"]
DATE_STYLE_PROBABILITIES = [0.50, 0.25, 0.10, 0.10, 0.05]


# ----------------------------------------------------------------------
# Step 1: build a tidy, realistic dataset
# ----------------------------------------------------------------------
def build_clean_base(rng, n):
    """Create n students whose marks depend (realistically) on their habits."""
    parental_education = rng.choice(config.EDUCATION_ORDER, size=n, p=[0.35, 0.20, 0.30, 0.15])
    school_type = rng.choice(config.SCHOOL_TYPES, size=n, p=[0.40, 0.20, 0.40])
    internet_access = rng.choice(["Yes", "No"], size=n, p=[0.75, 0.25])
    extra_classes = rng.choice(["Yes", "No"], size=n, p=[0.45, 0.55])

    study_hours = np.clip(rng.gamma(shape=5, scale=3.5, size=n), 1, 45)
    attendance = np.clip(rng.normal(85, 9, n), 50, 100)
    previous_score = np.clip(rng.normal(55, 12, n) + 0.4 * (study_hours - 17.5), 20, 100)

    education_bonus = {"High School": 0.0, "Diploma": 1.5, "Bachelor's": 3.0, "Master's": 4.5}
    school_bonus = {"Government": 0.0, "Aided": 1.0, "Private": 3.0}

    # One "ability" number per student that drives all three subjects
    common = (
        10
        + 0.9 * study_hours
        + 0.2 * (attendance - 50)
        + 0.35 * previous_score
        + np.array([education_bonus[e] for e in parental_education])
        + np.array([school_bonus[s] for s in school_type])
        + np.where(extra_classes == "Yes", 4, 0)
        + np.where(internet_access == "Yes", 2, 0)
    )
    math = common + rng.normal(0, 7, n)
    science = common + 1 + rng.normal(0, 7, n)
    english = common - 1 + rng.normal(0, 7, n)

    exam_dates = pd.Timestamp("2026-03-02") + pd.to_timedelta(rng.integers(0, 26, n), unit="D")

    df = pd.DataFrame({
        "student_id": [f"STU{1001 + i}" for i in range(n)],
        "student_name": [f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(n)],
        "gender": rng.choice(["Male", "Female"], size=n),
        "age": rng.choice([15, 16, 17, 18, 19], size=n, p=[0.15, 0.25, 0.30, 0.20, 0.10]).astype(float),
        "school_type": school_type,
        "parental_education": parental_education,
        "internet_access": internet_access,
        "extra_classes": extra_classes,
        "study_hours_per_week": study_hours.round(1),
        "attendance_percent": attendance.round(1),
        "previous_score": previous_score.round(1),
        "math_score": np.clip(math, 0, 100).round(1),
        "science_score": np.clip(science, 0, 100).round(1),
        "english_score": np.clip(english, 0, 100).round(1),
        "exam_date": exam_dates,
        "data_entry_operator": rng.choice(["OP-01", "OP-02", "OP-03"], size=n),
        "internal_remarks": rng.choice(
            ["", "", "", "Verified", "Re-checked", "Entered late", "Form scanned"], size=n),
        "record_source": "Survey_Form_2026",
    })
    return df


# ----------------------------------------------------------------------
# Step 2: damage the data in realistic ways
# ----------------------------------------------------------------------
def _pick_rows(rng, df, column, how_many):
    """Pick random row labels where `column` currently has a real value."""
    candidates = df.index[df[column].notna()].to_numpy()
    return rng.choice(candidates, size=how_many, replace=False)


def add_missing_values(df, rng, fraction=0.05):
    """Blank out about 5% of the cells in ten columns."""
    columns = [
        "age", "gender", "parental_education", "internet_access",
        "study_hours_per_week", "attendance_percent", "previous_score",
        "math_score", "science_score", "english_score",
    ]
    how_many = int(fraction * len(df))
    for column in columns:
        rows = rng.choice(len(df), size=how_many, replace=False)
        df.loc[rows, column] = np.nan
    return df


def add_invalid_and_outlier_values(df, rng):
    """Insert impossible values and extreme (outlier) values."""
    plan = {
        "age": [150, 5, -3],                                  # impossible
        "study_hours_per_week": [85, 95, 110, 120, 150],      # extreme but "possible"
        "attendance_percent": [105, 120, 150, 250],           # above 100 %
        "math_score": [150, 999, -20, 4, 8],                  # impossible + very low
        "science_score": [120, 105, 6],
        "english_score": [-5, 200, 3],
    }
    for column, values in plan.items():
        rows = _pick_rows(rng, df, column, len(values))
        df.loc[rows, column] = values
    return df


def _messy_text(value, rng):
    """Return the same text with random upper/lower case or extra spaces."""
    if not isinstance(value, str):
        return value
    options = [value.lower(), value.upper(), " " + value, value + " "]
    return options[rng.integers(0, len(options))]


def add_format_problems(df, rng):
    """Make labels, percentages and dates inconsistent."""
    n = len(df)

    def vary(value, variants):
        # 60% of the time keep the tidy label, otherwise use a variant
        if not isinstance(value, str) or rng.random() < 0.60:
            return value
        return str(rng.choice(variants[value]))

    df["gender"] = [vary(v, GENDER_VARIANTS) for v in df["gender"]]
    df["internet_access"] = [vary(v, YES_NO_VARIANTS) for v in df["internet_access"]]
    df["extra_classes"] = [vary(v, YES_NO_VARIANTS) for v in df["extra_classes"]]

    df["school_type"] = [_messy_text(v, rng) if rng.random() < 0.12 else v for v in df["school_type"]]
    df["parental_education"] = [
        _messy_text(v, rng) if rng.random() < 0.08 else v for v in df["parental_education"]]

    # 15% of attendance values are typed like "87.5%" (text instead of a number)
    use_percent = rng.random(n) < 0.15
    df["attendance_percent"] = [
        f"{v}%" if (flag and pd.notna(v)) else v
        for v, flag in zip(df["attendance_percent"], use_percent)
    ]

    # Dates typed in five different styles
    styles = rng.choice(len(DATE_STYLES), size=n, p=DATE_STYLE_PROBABILITIES)
    df["exam_date"] = [ts.strftime(DATE_STYLES[s]) for ts, s in zip(df["exam_date"], styles)]
    return df


def add_missing_tokens(df, rng):
    """Some people type '?', '-' or 'N/A' instead of leaving a cell empty."""
    tokens = {"math_score": ("?", 4), "study_hours_per_week": ("-", 3), "parental_education": ("N/A", 4)}
    for column, (token, how_many) in tokens.items():
        df[column] = df[column].astype(object)
        empty_rows = df.index[df[column].isna()].to_numpy()
        rows = rng.choice(empty_rows, size=min(how_many, len(empty_rows)), replace=False)
        df.loc[rows, column] = token
    return df


def add_duplicates(df, rng, seed):
    """Append 20 exact copies and 6 'near' copies (same student, retyped)."""
    picks = rng.choice(len(df), size=26, replace=False)
    exact_copies = df.iloc[picks[:20]].copy()
    near_copies = df.iloc[picks[20:]].copy()

    # A near copy is the same student typed again with different capitalisation
    for column in ["gender", "school_type"]:
        near_copies[column] = [v.swapcase() if isinstance(v, str) else v for v in near_copies[column]]
    near_copies["student_name"] = near_copies["student_name"].str.upper()

    combined = pd.concat([df, exact_copies, near_copies], ignore_index=True)
    # Shuffle so the duplicates are not all at the bottom
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


# ----------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------
def generate_raw_dataset(path=config.RAW_PATH, seed=SEED, n_students=N_STUDENTS):
    """Build the raw dataset, save it as CSV and return it."""
    rng = np.random.default_rng(seed)

    df = build_clean_base(rng, n_students)
    df = add_missing_values(df, rng)
    df = add_invalid_and_outlier_values(df, rng)
    df = add_format_problems(df, rng)
    df = add_missing_tokens(df, rng)
    df = add_duplicates(df, rng, seed)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    data = generate_raw_dataset()
    print(f"Synthetic raw dataset saved to: {config.RAW_PATH}")
    print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
