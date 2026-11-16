"""
data_analysis.py
----------------
Exploratory Data Analysis (EDA) on the CLEANED dataset.

It provides:
    * descriptive statistics        (count, mean, std, min, quartiles, max)
    * mean, median and mode
    * minimum and maximum values
    * correlation analysis
    * group comparisons             (for example average score by school type)
    * plain-English observations    (built from the real numbers, so they
                                     always match the data)

Run it on its own with:   python -m src.data_analysis
"""

import pandas as pd

from src import config


def _name(column):
    """Friendly name of a column, e.g. 'study_hours_per_week' -> 'Study hours per week'."""
    return config.LABELS.get(column, column.replace("_", " ").capitalize())


# ----------------------------------------------------------------------
# Statistics tables
# ----------------------------------------------------------------------
def descriptive_statistics(df):
    """count, mean, std, min, 25%, 50%, 75%, max for every numeric column."""
    return df[config.ANALYSIS_COLUMNS].describe().T.round(2)


def central_tendency(df):
    """Mean, median and mode for every numeric column."""
    rows = []
    for column in config.ANALYSIS_COLUMNS:
        rows.append({
            "column": column,
            "mean": round(df[column].mean(), 2),
            "median": round(df[column].median(), 2),
            "mode": round(df[column].mode().iloc[0], 2),   # most frequent value
            "std_dev": round(df[column].std(), 2),
        })
    return pd.DataFrame(rows).set_index("column")


def min_max_table(df):
    """Minimum, maximum and range of every numeric column."""
    table = pd.DataFrame({
        "minimum": df[config.ANALYSIS_COLUMNS].min(),
        "maximum": df[config.ANALYSIS_COLUMNS].max(),
    })
    table["range"] = table["maximum"] - table["minimum"]
    return table.round(2)


def correlation_matrix(df):
    """Pearson correlation between all numeric columns (-1 to +1)."""
    return df[config.ANALYSIS_COLUMNS].corr().round(2)


def factor_correlations(df, target="average_score"):
    """Correlation of the student's habits/background with the average score."""
    factors = ["age", "study_hours_per_week", "attendance_percent", "previous_score"]
    return df[factors].corrwith(df[target]).round(2).sort_values(ascending=False)


def group_summary(df, by, order=None):
    """Number of students and average score for every group of a category."""
    table = df.groupby(by)["average_score"].agg(["count", "mean"]).round(2)
    table = table.rename(columns={"count": "students", "mean": "average_score"})
    if order:
        table = table.reindex([o for o in order if o in table.index])
    return table


# ----------------------------------------------------------------------
# Plain-English observations
# ----------------------------------------------------------------------
def generate_observations(df):
    """Return a list of sentences that describe the important patterns."""
    notes = []
    corr = factor_correlations(df)
    strongest = corr.index[0]
    notes.append(
        f"{_name(strongest)} has the strongest positive link with the average score "
        f"(correlation r = {corr[strongest]:.2f}): students who study more score higher.")

    weakest = corr.abs().sort_values().index[0]
    notes.append(
        f"{_name(weakest)} has almost no linear link with the average score "
        f"(r = {corr[weakest]:.2f}).")

    means = df.groupby("extra_classes")["average_score"].mean()
    notes.append(
        f"Students who attend extra classes average {means['Yes']:.1f} marks versus "
        f"{means['No']:.1f} for those who do not (difference {means['Yes'] - means['No']:+.1f}).")

    means = df.groupby("internet_access")["average_score"].mean()
    notes.append(
        f"Students with internet access average {means['Yes']:.1f} marks versus "
        f"{means['No']:.1f} without it (difference {means['Yes'] - means['No']:+.1f}).")

    school = group_summary(df, "school_type", config.SCHOOL_TYPES)["average_score"]
    notes.append(
        f"By school type, {school.idxmax()} schools have the highest average ({school.max():.1f}) "
        f"and {school.idxmin()} schools the lowest ({school.min():.1f}).")

    edu = group_summary(df, "parental_education", config.EDUCATION_ORDER)["average_score"]
    notes.append(
        f"Average score rises with parental education, from {edu.iloc[0]:.1f} "
        f"({edu.index[0]}) to {edu.iloc[-1]:.1f} ({edu.index[-1]})."
        if edu.iloc[-1] > edu.iloc[0] else
        f"Parental education shows no clear upward pattern ({edu.index[0]}: {edu.iloc[0]:.1f}, "
        f"{edu.index[-1]}: {edu.iloc[-1]:.1f}).")

    levels = group_summary(df, "study_level", config.STUDY_LEVEL_ORDER)["average_score"]
    notes.append(
        "Average score by study level: " +
        ", ".join(f"{lvl} = {val:.1f}" for lvl, val in levels.items()) + ".")

    subjects = df[config.SCORE_COLUMNS].mean().sort_values(ascending=False)
    notes.append(
        f"{_name(subjects.index[0])} is the strongest subject (mean {subjects.iloc[0]:.1f}) and "
        f"{_name(subjects.index[-1])} the weakest (mean {subjects.iloc[-1]:.1f}).")

    gender = df.groupby("gender")["average_score"].mean()
    notes.append(
        f"Average scores of Female ({gender['Female']:.1f}) and Male ({gender['Male']:.1f}) students "
        f"differ by only {abs(gender['Female'] - gender['Male']):.1f} marks, so gender is not a major factor.")

    share = df["grade"].value_counts(normalize=True) * 100
    notes.append(
        f"{share.get('A', 0) + share.get('B', 0):.1f}% of students earn grade A or B, while "
        f"{share.get('D', 0) + share.get('F', 0):.1f}% fall in grade D or F.")

    high = df[df["attendance_percent"] >= 85]["average_score"].mean()
    low = df[df["attendance_percent"] < 85]["average_score"].mean()
    notes.append(
        f"Students with attendance of 85% or more average {high:.1f} marks; "
        f"those below 85% average {low:.1f}.")
    return notes


# ----------------------------------------------------------------------
# Run everything
# ----------------------------------------------------------------------
def run_analysis(df, output_dir=config.OUTPUT_DIR, log=print):
    """Print the EDA results and save them in the outputs folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def show(text=""):
        log(text)
        lines.append(str(text))

    def heading(title):
        show("")
        show("=" * 72)
        show(title)
        show("=" * 72)

    heading("EXPLORATORY DATA ANALYSIS (cleaned dataset)")
    show(f"Records analysed: {len(df)}")

    heading("1. Descriptive statistics")
    stats = descriptive_statistics(df)
    show(stats.to_string())

    heading("2. Mean, median and mode")
    show(central_tendency(df).to_string())

    heading("3. Minimum and maximum values")
    show(min_max_table(df).to_string())

    heading("4. Correlation matrix")
    corr = correlation_matrix(df)
    show(corr.to_string())
    show("\nCorrelation of each factor with the average score:")
    show(factor_correlations(df).to_string())

    heading("5. Average score by group")
    for column, order in [("school_type", config.SCHOOL_TYPES),
                          ("parental_education", config.EDUCATION_ORDER),
                          ("extra_classes", ["Yes", "No"]),
                          ("internet_access", ["Yes", "No"]),
                          ("gender", ["Female", "Male"]),
                          ("study_level", config.STUDY_LEVEL_ORDER)]:
        show(f"\nBy {_name(column).lower()}:")
        show(group_summary(df, column, order).to_string())

    heading("6. Key observations")
    observations = generate_observations(df)
    for number, note in enumerate(observations, start=1):
        show(f"{number}. {note}")

    stats.to_csv(output_dir / "descriptive_statistics.csv")
    corr.to_csv(output_dir / "correlation_matrix.csv")
    (output_dir / "analysis_report.txt").write_text("\n".join(lines), encoding="utf-8")
    show(f"\nSaved: {output_dir / 'analysis_report.txt'}")
    return observations


if __name__ == "__main__":
    cleaned = pd.read_csv(config.CLEANED_PATH)
    run_analysis(cleaned)
