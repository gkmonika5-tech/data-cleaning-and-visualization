# Testing and sample outputs

## 1. How to run the tests

```
python -m pytest -v
```

## 2. Test checklist and results

| Requirement | How it is tested | Result |
|---|---|---|
| All Python files execute without errors | `main.py` run end-to-end; every file compiled; modules run with `python -m` | Passed |
| Missing values handled correctly | `test_no_missing_values_after_cleaning`, `test_missing_values_were_filled_with_median_or_mode` | Passed |
| Duplicate records removed | `test_duplicates_are_removed` (526 → 500 rows, unique IDs) | Passed |
| Formats corrected | `test_labels_are_standardised`, `test_numeric_columns_are_numbers_inside_valid_ranges`, `test_exam_date_is_a_real_date` | Passed |
| Outliers handled | `test_outliers_are_handled` (56 detected, 0 remaining, max study hours < 45) | Passed |
| Unnecessary columns removed | `test_unnecessary_columns_are_removed` | Passed |
| Cleaned CSV saved | `test_cleaned_csv_is_saved_and_can_be_reloaded` | Passed |
| Analysis works | `test_analysis_tables_and_observations` | Passed |
| Graphs generated | `test_all_graphs_are_created` (7 non-empty PNG files) | Passed |
| Dashboard runs correctly | See section 4 | See note |

Result of the last run: **14 passed**.

## 3. Sample output (from `outputs/run_log.txt`)

```
STEP 4: Identify missing values (raw data)
              column  missing_count  missing_percent
    internal_remarks            231            43.92
study_hours_per_week             28             5.32
...
Total missing cells: 495

STEP 6a: Identify and remove exact duplicate records
Exact duplicate rows found and removed: 20

STEP 8: Correct inconsistent data formats
  gender: 11 different spellings -> 2 standard labels
  exam_date: 5 different date styles -> one date format (YYYY-MM-DD)

STEP 7: Detect and handle outliers (IQR method, values are capped)
  study_hours_per_week: limits [-3.0, 37.8] -> 18 outlier(s) capped
```

**Explanation:** the first block shows the raw data has 495 empty cells (almost half are the unnecessary remarks column). The duplicate step removes 20 exact copies. The format step shows 11 spellings of gender collapse into 2 labels. The outlier step shows that any study-hours value above 37.8 per week is capped at 37.8.

Before and after cleaning:

| Measure | Before | After |
|---|---|---|
| Rows × columns | 526 × 18 | 500 × 17 |
| Missing cells | 495 | 0 |
| Duplicates (rows / repeated IDs) | 20 / 26 | 0 / 0 |
| Impossible values | 14 | 0 |
| IQR outliers | 56 | 0 |

## 4. Dashboard testing

`app.py` was exercised with a stand-in ("mock") for the Streamlit library: every line of the script ran without errors with (a) default filters, (b) a narrow filter selection and (c) an empty selection (which shows the "No students match" warning). This verifies the Python logic and that all charts are real Matplotlib figures.

It does **not** verify how the page looks in a browser. Please do the visual check yourself:

1. `streamlit run app.py`
2. Open each of the five tabs.
3. Change a sidebar filter and confirm the numbers and charts update.
4. Take screenshots of each tab and paste them into section 12 of the report.
