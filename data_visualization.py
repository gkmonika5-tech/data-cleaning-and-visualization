"""
data_visualization.py
---------------------
Creates the charts with Matplotlib and Seaborn.

Charts (all saved as PNG files in outputs/graphs/):
    1. Bar chart          - average score by parental education
    2. Pie chart          - share of students in each grade
    3. Histogram          - distribution of the average score
    4. Box plot           - average score by school type
    5. Scatter plot       - study hours vs average score
    6. Correlation heatmap
    7. Before/after box plot - effect of outlier handling (extra)

Every plot_* function RETURNS a Matplotlib figure, so the Streamlit dashboard
can show the very same charts with st.pyplot(fig).

Colours: one blue for single-series charts, an ordered set of distinct colours
for categories, and a blue-grey-red scale for correlation (positive/negative).

Run it on its own with:   python -m src.data_visualization
"""

import matplotlib

matplotlib.use("Agg")   # draw to files, no pop-up window needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src import config
from src.data_cleaning import to_number

# ----------------------------------------------------------------------
# Look and feel
# ----------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical colours in a fixed order: blue, orange, aqua, yellow, magenta
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# Diverging scale for correlations: red (negative) - grey (zero) - blue (positive)
CORR_CMAP = LinearSegmentedColormap.from_list("corr", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_SOFT, "axes.titleweight": "bold",
    "axes.titlesize": 14, "axes.titlecolor": INK, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "legend.frameon": False, "font.size": 10,
})

FIGURE_SIZE = (9, 5.5)


def _label(column):
    return config.LABELS.get(column, column)


def _style_legend(ax, **kwargs):
    legend = ax.legend(**kwargs)
    if legend:
        for text in legend.get_texts():
            text.set_color(INK_SOFT)
    return legend


# ----------------------------------------------------------------------
# 1. Bar chart
# ----------------------------------------------------------------------
def plot_bar_chart(df):
    """Average score for each level of parental education."""
    summary = (df.groupby("parental_education")["average_score"].mean()
               .reindex([e for e in config.EDUCATION_ORDER if e in df["parental_education"].unique()]))
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    bars = ax.bar(summary.index, summary.values, color=SERIES[0], width=0.55)
    ax.bar_label(bars, fmt="%.1f", padding=3, color=INK_SOFT)
    ax.set_title("Average Score by Parental Education")
    ax.set_xlabel("Parental education level")
    ax.set_ylabel("Average score (out of 100)")
    ax.set_ylim(0, max(100, summary.max() * 1.15) if len(summary) else 100)
    ax.grid(axis="x", visible=False)
    return fig


# ----------------------------------------------------------------------
# 2. Pie chart
# ----------------------------------------------------------------------
def plot_pie_chart(df):
    """Share of students in each grade."""
    counts = df["grade"].value_counts().reindex(config.GRADE_ORDER).dropna()
    colours = [SERIES[config.GRADE_ORDER.index(g)] for g in counts.index]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    wedges, _, autotexts = ax.pie(
        counts.values, colors=colours, startangle=90, counterclock=False,
        autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",
        pctdistance=0.72, wedgeprops={"edgecolor": SURFACE, "linewidth": 2})
    for text, grade in zip(autotexts, counts.index):
        text.set_color(INK if grade == "D" else "white")   # dark text on yellow
        text.set_fontweight("bold")
    ax.legend(wedges, [f"Grade {g} ({n} students)" for g, n in counts.items()],
              title="Grade", loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title("Distribution of Student Grades")
    ax.grid(False)
    return fig


# ----------------------------------------------------------------------
# 3. Histogram
# ----------------------------------------------------------------------
def plot_histogram(df):
    """How the average scores are spread out."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.histplot(df["average_score"], bins=20, kde=True, color=SERIES[0],
                 edgecolor=SURFACE, ax=ax)
    ax.axvline(df["average_score"].mean(), color=SERIES[1], linewidth=2,
               label=f"Mean = {df['average_score'].mean():.1f}")
    ax.axvline(df["average_score"].median(), color=INK_SOFT, linewidth=2, linestyle="--",
               label=f"Median = {df['average_score'].median():.1f}")
    ax.set_title("Distribution of Average Scores")
    ax.set_xlabel("Average score (out of 100)")
    ax.set_ylabel("Number of students")
    ax.grid(axis="x", visible=False)
    _style_legend(ax)
    return fig


# ----------------------------------------------------------------------
# 4. Box plot
# ----------------------------------------------------------------------
def plot_box_plot(df):
    """Spread of the average score for each school type."""
    order = [s for s in config.SCHOOL_TYPES if s in df["school_type"].unique()]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.boxplot(data=df, x="school_type", y="average_score", order=order, color=SERIES[0],
                linewidth=1.3, fliersize=4, ax=ax, boxprops={"alpha": 0.85})
    ax.set_title("Average Score by School Type")
    ax.set_xlabel("School type")
    ax.set_ylabel("Average score (out of 100)")
    ax.grid(axis="x", visible=False)
    return fig


# ----------------------------------------------------------------------
# 5. Scatter plot
# ----------------------------------------------------------------------
def plot_scatter(df):
    """Study hours against the average score, coloured by extra classes."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.scatterplot(data=df, x="study_hours_per_week", y="average_score",
                    hue="extra_classes", hue_order=["Yes", "No"],
                    palette={"Yes": SERIES[0], "No": SERIES[1]},
                    s=45, alpha=0.8, edgecolor=SURFACE, linewidth=0.8, ax=ax)
    if len(df) > 2:
        slope, intercept = np.polyfit(df["study_hours_per_week"], df["average_score"], 1)
        r = df["study_hours_per_week"].corr(df["average_score"])
        xs = np.linspace(df["study_hours_per_week"].min(), df["study_hours_per_week"].max(), 50)
        ax.plot(xs, slope * xs + intercept, color=INK_SOFT, linestyle="--", linewidth=2,
                label=f"Trend line (r = {r:.2f})")
    ax.set_title("Study Hours vs Average Score")
    ax.set_xlabel("Study hours per week")
    ax.set_ylabel("Average score (out of 100)")
    _style_legend(ax, title="Extra classes")
    return fig


# ----------------------------------------------------------------------
# 6. Correlation heatmap
# ----------------------------------------------------------------------
def plot_heatmap(df):
    """Correlation between all numeric columns."""
    corr = df[config.ANALYSIS_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=CORR_CMAP, vmin=-1, vmax=1, center=0,
                linewidths=2, linecolor=SURFACE, square=True,
                xticklabels=[_label(c) for c in corr.columns],
                yticklabels=[_label(c) for c in corr.columns],
                cbar_kws={"label": "Correlation (r)", "shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Columns")
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    return fig


# ----------------------------------------------------------------------
# 7. Before / after outlier handling (extra chart)
# ----------------------------------------------------------------------
def plot_outlier_before_after(raw_df, clean_df, column="study_hours_per_week"):
    """Box plot of one column before and after cleaning."""
    before = to_number(raw_df[column]).dropna()
    data = pd.DataFrame({
        "Stage": ["Before cleaning"] * len(before) + ["After cleaning"] * len(clean_df),
        "value": list(before) + list(clean_df[column]),
    })
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.boxplot(data=data, x="Stage", y="value", color=SERIES[0], linewidth=1.3,
                fliersize=4, ax=ax, boxprops={"alpha": 0.85})
    ax.set_title(f"{_label(column)}: Before vs After Outlier Handling")
    ax.set_xlabel("")
    ax.set_ylabel(_label(column))
    ax.grid(axis="x", visible=False)
    return fig


# ----------------------------------------------------------------------
# Explanations ("what does each chart tell us?")
# ----------------------------------------------------------------------
def explain_graphs(df, raw_df=None):
    """Return {chart name: explanation}. Numbers come from the data itself."""
    edu = df.groupby("parental_education")["average_score"].mean()
    edu = edu.reindex([e for e in config.EDUCATION_ORDER if e in edu.index])
    grades = df["grade"].value_counts(normalize=True) * 100
    school = df.groupby("school_type")["average_score"].median()
    r = df["study_hours_per_week"].corr(df["average_score"])
    corr = df[config.ANALYSIS_COLUMNS].corr()["average_score"].drop(
        ["average_score", "math_score", "science_score", "english_score"]).sort_values(ascending=False)

    text = {
        "Bar chart": (
            f"Compares the mean average score of students by their parents' education. "
            f"Scores go from {edu.iloc[0]:.1f} ({edu.index[0]}) to {edu.iloc[-1]:.1f} ({edu.index[-1]}), "
            f"so a better-educated home is linked with somewhat higher marks."),
        "Pie chart": (
            f"Shows how the {len(df)} students are spread over grades. Grades B and C are the most "
            f"common ({grades.get('B', 0):.1f}% and {grades.get('C', 0):.1f}%); only "
            f"{grades.get('A', 0):.1f}% reach grade A and {grades.get('F', 0):.1f}% get grade F."),
        "Histogram": (
            f"Shows the shape of the scores. The mean ({df['average_score'].mean():.1f}) and the median "
            f"({df['average_score'].median():.1f}) are close, so the distribution is roughly "
            f"bell-shaped and symmetric, with most students between "
            f"{df['average_score'].quantile(0.25):.0f} and {df['average_score'].quantile(0.75):.0f} marks."),
        "Box plot": (
            f"Compares the spread of scores for each school type. The box is the middle 50% of "
            f"students and the line inside is the median. {school.idxmax()} schools have the highest "
            f"median ({school.max():.1f}) and {school.idxmin()} the lowest ({school.min():.1f}); dots "
            f"outside the whiskers are unusual students."),
        "Scatter plot": (
            f"Each dot is a student. The points rise from left to right: more weekly study hours go "
            f"with higher scores (correlation r = {r:.2f}, a strong positive relationship). "
            f"Students with extra classes tend to sit above those without."),
        "Correlation heatmap": (
            f"Each cell is the correlation between two columns: +1 means they rise together, 0 means "
            f"no relationship, -1 means one rises as the other falls. Study hours "
            f"(r = {corr.get('study_hours_per_week', float('nan')):.2f}) and previous score "
            f"(r = {corr.get('previous_score', float('nan')):.2f}) are linked with the average score; "
            f"age has almost no link. The three subject scores are strongly related to each other."),
    }
    if raw_df is not None:
        before = to_number(raw_df["study_hours_per_week"])
        text["Before/after box plot"] = (
            f"Before cleaning, study hours reach {before.max():.0f} per week (impossible for a student); "
            f"after cleaning the maximum is {df['study_hours_per_week'].max():.1f}. The extreme values "
            f"were capped with the IQR rule, so the box plot is no longer stretched by outliers.")
    return text


# ----------------------------------------------------------------------
# Save everything
# ----------------------------------------------------------------------
def save_figure(fig, name, output_dir):
    """Save a figure as PNG, close it and return the file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def create_all_graphs(df, raw_df=None, output_dir=config.GRAPH_DIR, log=print):
    """Create and save all charts. Returns {file name: path}."""
    plots = [
        ("01_bar_chart.png", plot_bar_chart(df)),
        ("02_pie_chart.png", plot_pie_chart(df)),
        ("03_histogram.png", plot_histogram(df)),
        ("04_box_plot.png", plot_box_plot(df)),
        ("05_scatter_plot.png", plot_scatter(df)),
        ("06_correlation_heatmap.png", plot_heatmap(df)),
    ]
    if raw_df is not None:
        plots.append(("07_outliers_before_after.png", plot_outlier_before_after(raw_df, df)))

    saved = {}
    for file_name, figure in plots:
        saved[file_name] = save_figure(figure, file_name, output_dir)
        log(f"  saved {saved[file_name]}")

    # Write the explanations next to the pictures
    lines = ["# What each graph tells us", ""]
    for title, explanation in explain_graphs(df, raw_df).items():
        lines += [f"## {title}", explanation, ""]
    (output_dir.parent / "graph_explanations.md").write_text("\n".join(lines), encoding="utf-8")
    return saved


if __name__ == "__main__":
    from src.data_cleaning import load_raw_data

    cleaned = pd.read_csv(config.CLEANED_PATH)
    create_all_graphs(cleaned, load_raw_data())
