# Explaining the project to your faculty (simple version)

## The one-minute story

"Real data is messy. My project takes a messy student-performance dataset with 526 records, cleans it with Python and Pandas, analyses it, draws charts, and shows everything in an interactive dashboard. After cleaning, I have 500 reliable records and I can see which factors are linked with marks — study hours matter most."

## Why I used a synthetic dataset

I generated the data with Python (`src/generate_dataset.py`) so that it contains *every* problem the project needs to show: missing values, duplicates, impossible values, outliers, inconsistent formats and useless columns. It is clearly marked synthetic — no real student is involved. The same code can be pointed at a real CSV with the same columns.

## What I did, in five steps

1. **Raw data** – 526 rows, 18 columns, full of problems.
2. **Cleaning** – fixed each problem (details below).
3. **Analysis (EDA)** – statistics and correlations.
4. **Visualization** – six required charts plus one before/after chart.
5. **Dashboard** – Streamlit app with filters and interactive charts.

## The cleaning problems and how I solved them

| Problem | Example | Solution |
|---|---|---|
| Missing values | Empty math score | Fill with the **median** (numbers) or **mode** (categories) |
| Duplicates | Same student entered twice | `drop_duplicates()` — 26 removed |
| Impossible values | Age 150, marks 999 | Treated as missing, then filled |
| Outliers | 120 study hours/week | **IQR rule**, values capped at the limit |
| Inconsistent formats | `M`, `male`, `MALE`; `87%`; 5 date styles | Mapped to one label / number / date format |
| Unnecessary columns | Name, operator, remarks | Dropped |

## Simple definitions you may be asked

- **Missing value** – a cell with no data.
- **Median** – the middle value when sorted; not affected by extremes. **Mode** – the most frequent value.
- **Outlier** – a value far away from the rest.
- **IQR rule** – Q1 − 1.5×IQR to Q3 + 1.5×IQR is the "normal" range; anything outside is an outlier.
- **Correlation** – a number from −1 to +1 showing how two columns move together (0 = no relationship).
- **EDA** – Exploratory Data Analysis: looking at the data with statistics and charts before drawing conclusions.

## What the results show

- Study hours ↔ average score: **r = 0.71** (strong positive).
- Previous exam score ↔ average score: **r = 0.58**.
- Students with *High* study level average 66.9 marks; *Low* study level 48.9.
- Extra classes +3.1 marks, internet access +3.0, private schools +4 marks over government schools.
- Age, attendance and gender show almost no link.

## Likely questions and good answers

**Why did you cap outliers instead of deleting them?** Deleting the row would lose the student's other valid data; capping keeps the record but limits the extreme value.

**Why median instead of mean for filling?** Extreme values distort the mean; the median is robust.

**Why is the mode equal to the median in the statistics?** About 25 missing cells per column were filled with the median, so it became the most frequent value. That is a known side effect of simple imputation.

**Why is the cleaning order not 1 to 10?** Formats must be fixed before duplicates and missing values can be judged, and outliers are checked after filling. The code labels each step with its number.

**Does this prove that study hours cause higher marks?** No. Correlation is not causation, and the data is synthetic.

**How do you know the cleaning worked?** 14 automated tests (`python -m pytest -v`) check no missing values, no duplicates, valid ranges, standard labels, a saved CSV and all graphs.

**What are the limitations?** Synthetic data; simple filling reduces natural variation; no predictive model yet.

**What can be improved?** Use a real dataset, fill missing values by group, add a regression/machine-learning model, deploy the dashboard online.

## Suggested 5-minute demo

1. Open `dataset/raw_data.csv` and point out a few problems.
2. Run `python main.py` and scroll through the printed steps and the before/after table.
3. Show `dataset/cleaned_data.csv` and the graphs in `outputs/graphs/`.
4. Run `streamlit run app.py`: show *Data Quality*, then use the sidebar filters in *Explore*.
5. Finish with the *Findings* tab and the limitations.
