"""
app.py  -  Interactive Streamlit dashboard
==========================================
Run it with:

    streamlit run app.py

The dashboard has five tabs:
    1. Overview        - title, introduction, dataset preview, column dictionary
    2. Data Quality    - missing values, duplicates, outliers, before/after
    3. Explore         - sidebar filters + interactive charts + download
    4. Charts          - the Matplotlib / Seaborn charts for the filtered data
    5. Findings        - statistics, key findings and conclusion

The dashboard cleans the raw CSV itself (using src/data_cleaning.py), so it
always shows results that match the code. If dataset/raw_data.csv is missing
it is generated automatically.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src import data_visualization as viz
from src.data_analysis import (central_tendency, correlation_matrix, descriptive_statistics,
                               factor_correlations, generate_observations)
from src.data_cleaning import build_comparison_table, clean_data, load_raw_data
from src.generate_dataset import generate_raw_dataset

# ----------------------------------------------------------------------
# Page set-up (must be the first Streamlit command)
# ----------------------------------------------------------------------
st.set_page_config(page_title="Student Data Cleaning & Visualization",
                   page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {
        background-color: rgba(42, 120, 214, 0.08);
        border: 1px solid rgba(42, 120, 214, 0.25);
        border-radius: 10px; padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

LABEL = config.LABELS
CATEGORY_ORDER = {
    "school_type": config.SCHOOL_TYPES,
    "parental_education": config.EDUCATION_ORDER,
    "grade": config.GRADE_ORDER,
    "study_level": config.STUDY_LEVEL_ORDER,
    "gender": ["Female", "Male"],
    "internet_access": ["Yes", "No"],
    "extra_classes": ["Yes", "No"],
}


# ----------------------------------------------------------------------
# Load and clean the data (cached, so it only runs once)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning the dataset ...")
def load_everything():
    if not config.RAW_PATH.exists():
        generate_raw_dataset()
    raw = load_raw_data()
    cleaned, report = clean_data(raw)
    return raw, cleaned, report


raw_df, df, report = load_everything()


def show_figure(fig):
    """Display a Matplotlib figure and free its memory."""
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------
# Sidebar filters (used by the Explore and Charts tabs)
# ----------------------------------------------------------------------
st.sidebar.title("Filters")
st.sidebar.caption("The filters change the **Explore** and **Charts** tabs. "
                   "Refresh the page to reset them.")


def ordered_options(column):
    present = set(df[column].unique())
    return [v for v in CATEGORY_ORDER[column] if v in present]


gender_sel = st.sidebar.multiselect("Gender", ordered_options("gender"), default=ordered_options("gender"))
school_sel = st.sidebar.multiselect("School type", ordered_options("school_type"),
                                    default=ordered_options("school_type"))
edu_sel = st.sidebar.multiselect("Parental education", ordered_options("parental_education"),
                                 default=ordered_options("parental_education"))
internet_sel = st.sidebar.multiselect("Internet access", ordered_options("internet_access"),
                                      default=ordered_options("internet_access"))
extra_sel = st.sidebar.multiselect("Extra classes", ordered_options("extra_classes"),
                                   default=ordered_options("extra_classes"))
grade_sel = st.sidebar.multiselect("Grade", ordered_options("grade"), default=ordered_options("grade"))
age_sel = st.sidebar.slider("Age", int(df["age"].min()), int(df["age"].max()),
                            (int(df["age"].min()), int(df["age"].max())))
hours_min, hours_max = float(df["study_hours_per_week"].min()), float(df["study_hours_per_week"].max())
hours_sel = st.sidebar.slider("Study hours per week", hours_min, hours_max, (hours_min, hours_max), step=0.5)

mask = (
    df["gender"].isin(gender_sel)
    & df["school_type"].isin(school_sel)
    & df["parental_education"].isin(edu_sel)
    & df["internet_access"].isin(internet_sel)
    & df["extra_classes"].isin(extra_sel)
    & df["grade"].isin(grade_sel)
    & df["age"].between(age_sel[0], age_sel[1])
    & df["study_hours_per_week"].between(hours_sel[0], hours_sel[1])
)
filtered = df[mask]

st.sidebar.divider()
st.sidebar.metric("Students matching filters", f"{len(filtered)} of {len(df)}")
st.sidebar.caption("Dataset: synthetic (computer-generated) student data. No real person is described.")


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("📊 Student Performance: Data Cleaning & Visualization")
st.markdown(
    "This project takes a **raw, messy dataset** of student records and turns it into **clean, "
    "meaningful information**. Use the tabs below to see the data before and after cleaning, "
    "explore it with filters and charts, and read the key findings."
)

tab_overview, tab_quality, tab_explore, tab_charts, tab_findings = st.tabs(
    ["🏠 Overview", "🧹 Data Quality", "🔎 Explore", "📈 Charts", "💡 Findings"])


# ======================================================================
# TAB 1 - Overview
# ======================================================================
with tab_overview:
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records (raw)", report["rows_before"])
    c2.metric("Records after cleaning", report["rows_after"])
    c3.metric("Columns (raw → clean)", f"{report['cols_before']} → {report['cols_after']}")
    c4.metric("Missing cells (raw → clean)",
              f"{report['missing_before_total']} → {report['missing_after_total']}")

    st.markdown(
        "**About the data:** the dataset is **synthetic**. It was generated by "
        "`src/generate_dataset.py` and deliberately contains missing values, duplicate records, "
        "impossible values, outliers, inconsistent formats and unnecessary columns, so that every "
        "cleaning step has something real to fix.")

    st.subheader("Preview")
    view = st.radio("Choose the dataset to preview", ["Raw data (before cleaning)", "Cleaned data (after cleaning)"],
                    horizontal=True)
    if view.startswith("Raw"):
        st.caption(f"Raw data: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns (first 10 rows)")
        st.dataframe(raw_df.head(10))
    else:
        st.caption(f"Cleaned data: {df.shape[0]} rows × {df.shape[1]} columns (first 10 rows)")
        st.dataframe(df.head(10))

    st.subheader("Column dictionary")
    dictionary = pd.DataFrame(config.COLUMN_INFO, columns=["Column", "Type", "Description"])
    dictionary["In cleaned data?"] = np.where(dictionary["Column"].isin(df.columns), "Yes", "Removed")
    st.dataframe(dictionary, hide_index=True)


# ======================================================================
# TAB 2 - Data Quality
# ======================================================================
with tab_quality:
    st.subheader("Missing values summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Missing cells before", report["missing_before_total"])
    m2.metric("Rows with a missing value", report["rows_with_missing_before"])
    m3.metric("Missing cells after", report["missing_after_total"])

    missing_series = pd.Series(report["missing_before"])
    missing_series = missing_series[missing_series > 0].sort_values(ascending=False)
    st.caption("Missing values per column in the raw data "
               "(internal_remarks is a free-text column that is removed later).")
    st.bar_chart(missing_series)

    imputation = pd.DataFrame(report["imputation"]).T.reset_index()
    imputation.columns = ["Column", "Method", "Value used", "Cells filled"]
    imputation["Value used"] = imputation["Value used"].astype(str)
    imputation["Cells filled"] = imputation["Cells filled"].astype(int)
    with st.expander("How the missing values were filled", expanded=False):
        st.markdown("Numeric columns use the **median** (not affected by extreme values). "
                    "Category columns use the **mode** (most frequent label).")
        st.dataframe(imputation, hide_index=True)

    st.subheader("Duplicate records summary")
    d1, d2, d3 = st.columns(3)
    d1.metric("Exact duplicate rows", report["exact_duplicates"])
    d2.metric("Re-typed duplicates (same student ID)", report["near_duplicates"])
    d3.metric("Total duplicates removed", report["duplicates_removed_total"])
    with st.expander("Show examples of duplicated student IDs in the raw data"):
        repeated = raw_df[raw_df["student_id"].duplicated(keep=False)].sort_values("student_id")
        st.dataframe(repeated[["student_id", "gender", "age", "school_type", "math_score",
                               "science_score", "english_score"]].head(12), hide_index=True)

    st.subheader("Impossible values and outliers")
    rows = []
    for column in config.OUTLIER_COLUMNS + ["age"]:
        outlier = report["outliers"].get(column)
        rows.append({
            "Column": LABEL[column],
            "Impossible values set to missing": report["invalid_values"].get(column, 0),
            "IQR lower limit": outlier["lower"] if outlier else None,
            "IQR upper limit": outlier["upper"] if outlier else None,
            "Outliers capped": outlier["count"] if outlier else 0,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    chosen = st.selectbox("See the effect of outlier handling for:", config.OUTLIER_COLUMNS,
                          format_func=lambda c: LABEL[c])
    show_figure(viz.plot_outlier_before_after(raw_df, df, chosen))

    st.subheader("Before-and-after cleaning comparison")
    st.dataframe(build_comparison_table(report), hide_index=True)

    with st.expander("Formatting problems that were corrected"):
        st.markdown(
            "- **Gender / Yes-No labels**: `male`, `M`, `MALE` ... → one standard label.\n"
            "- **School type / parental education**: different capital letters and extra spaces → standard names.\n"
            "- **Attendance**: values such as `87.5%` (text) → number `87.5`.\n"
            "- **Exam date**: five different date styles → one real date (YYYY-MM-DD).\n"
            "- **Age**: stored as a whole number.")


# ======================================================================
# TAB 3 - Explore (interactive charts)
# ======================================================================
with tab_explore:
    st.subheader("Explore the cleaned data")
    if filtered.empty:
        st.warning("No students match the current filters. Please widen the filters in the sidebar.")
    else:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Students shown", len(filtered))
        e2.metric("Average score", f"{filtered['average_score'].mean():.1f}")
        e3.metric("Average study hours / week", f"{filtered['study_hours_per_week'].mean():.1f}")
        e4.metric("Average attendance", f"{filtered['attendance_percent'].mean():.1f}%")

        left, right = st.columns(2)
        with left:
            st.markdown("**Average of a measure for each group**")
            group_col = st.selectbox("Group by", list(CATEGORY_ORDER), format_func=lambda c: LABEL[c])
            measure = st.selectbox("Measure", ["average_score", "math_score", "science_score", "english_score",
                                               "study_hours_per_week", "attendance_percent", "previous_score"],
                                   format_func=lambda c: LABEL[c])
            summary = filtered.groupby(group_col)[measure].mean().round(2)
            summary = summary.reindex([v for v in CATEGORY_ORDER[group_col] if v in summary.index])
            st.bar_chart(summary)
        with right:
            st.markdown("**Distribution of a column**")
            hist_col = st.selectbox("Column", config.ANALYSIS_COLUMNS, index=len(config.ANALYSIS_COLUMNS) - 1,
                                    format_func=lambda c: LABEL[c])
            bins = st.slider("Number of bins", 5, 40, 15)
            counts, edges = np.histogram(filtered[hist_col], bins=bins)
            middles = ((edges[:-1] + edges[1:]) / 2).round(1)
            st.bar_chart(pd.DataFrame({"Number of students": counts}, index=middles))

        st.markdown("**Relationship between two columns** (hover over a dot to see its values)")
        s1, s2, s3 = st.columns(3)
        x_col = s1.selectbox("X axis", config.ANALYSIS_COLUMNS, index=1, format_func=lambda c: LABEL[c])
        y_col = s2.selectbox("Y axis", config.ANALYSIS_COLUMNS, index=len(config.ANALYSIS_COLUMNS) - 1,
                             format_func=lambda c: LABEL[c])
        color_col = s3.selectbox("Colour by", ["extra_classes", "school_type", "internet_access", "gender", "grade"],
                                 format_func=lambda c: LABEL[c])
        st.scatter_chart(filtered, x=x_col, y=y_col, color=color_col)

        st.markdown("**Filtered data**")
        st.dataframe(filtered)
        st.download_button("Download filtered data as CSV", filtered.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_students.csv", mime="text/csv")


# ======================================================================
# TAB 4 - Charts (Matplotlib / Seaborn)
# ======================================================================
with tab_charts:
    st.subheader("Visualizations (for the filtered students)")
    if len(filtered) < 10:
        st.warning("Please select at least 10 students (use the sidebar filters) to draw these charts.")
    else:
        explanations = viz.explain_graphs(filtered)
        charts = [
            ("Bar chart", viz.plot_bar_chart),
            ("Pie chart", viz.plot_pie_chart),
            ("Histogram", viz.plot_histogram),
            ("Box plot", viz.plot_box_plot),
            ("Scatter plot", viz.plot_scatter),
            ("Correlation heatmap", viz.plot_heatmap),
        ]
        for title, function in charts:
            st.markdown(f"### {title}")
            chart_col, text_col = st.columns([3, 2])
            with chart_col:
                show_figure(function(filtered))
            with text_col:
                st.markdown("**What this tells us**")
                st.write(explanations[title])
            st.divider()


# ======================================================================
# TAB 5 - Findings
# ======================================================================
with tab_findings:
    st.subheader("Key findings")
    st.caption("Based on all cleaned records (filters do not change this tab).")
    for number, note in enumerate(generate_observations(df), start=1):
        st.markdown(f"{number}. {note}")

    st.subheader("Statistics")
    st.markdown("**Descriptive statistics**")
    st.dataframe(descriptive_statistics(df))
    st.markdown("**Mean, median and mode**")
    st.dataframe(central_tendency(df))
    st.markdown("**Correlation matrix**")
    st.dataframe(correlation_matrix(df))

    factors = factor_correlations(df)
    st.subheader("Conclusion")
    st.success(
        f"Cleaning turned {report['rows_before']} messy rows into {report['rows_after']} reliable records "
        f"with no missing values, no duplicates and no impossible values. The clean data shows that "
        f"**weekly study hours** ({factors['study_hours_per_week']:.2f}) and **previous exam results** "
        f"({factors['previous_score']:.2f}) are the strongest factors linked with the average score, "
        f"while age and attendance have almost no linear relationship with it in this dataset.")
    st.info("Note: the data is synthetic, so these relationships illustrate the method; "
            "they are not conclusions about real students. Correlation does not prove cause and effect.")
