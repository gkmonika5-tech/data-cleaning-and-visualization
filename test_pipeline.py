"""
Automated tests for the project.

Run them from the project folder with:

    python -m pytest -v

All files are created inside a temporary folder, so your real dataset and
graphs are never touched by the tests.
"""

import pandas as pd
import pytest

from src import config
from src.data_analysis import (central_tendency, correlation_matrix, descriptive_statistics,
                               generate_observations, run_analysis)
from src.data_cleaning import build_comparison_table, load_raw_data, run_cleaning
from src.data_visualization import create_all_graphs
from src.generate_dataset import generate_raw_dataset


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """Generate raw data in a temp folder, clean it and return everything."""
    folder = tmp_path_factory.mktemp("project")
    raw_path = folder / "raw_data.csv"
    cleaned_path = folder / "cleaned_data.csv"
    report_path = folder / "cleaning_report.json"

    generate_raw_dataset(raw_path)
    raw, cleaned, report = run_cleaning(raw_path, cleaned_path, report_path, log=lambda *_: None)
    return {"folder": folder, "raw": raw, "cleaned": cleaned, "report": report,
            "cleaned_path": cleaned_path, "report_path": report_path}


# ---------------- the raw data really is messy ----------------
def test_raw_data_has_the_planned_problems(project):
    raw = project["raw"]
    assert raw.isna().sum().sum() > 0                     # missing values
    assert raw.duplicated().sum() == 20                   # exact duplicates
    assert raw["student_id"].duplicated().sum() == 26     # exact + re-typed duplicates
    assert raw["gender"].nunique() > 2                    # inconsistent labels
    assert any(c in raw.columns for c in config.COLUMNS_TO_DROP)


# ---------------- missing values ----------------
def test_no_missing_values_after_cleaning(project):
    assert project["cleaned"].isna().sum().sum() == 0


def test_missing_values_were_filled_with_median_or_mode(project):
    imputation = project["report"]["imputation"]
    assert imputation["math_score"]["method"] == "median"
    assert imputation["gender"]["method"] == "mode"


# ---------------- duplicates ----------------
def test_duplicates_are_removed(project):
    cleaned = project["cleaned"]
    assert cleaned.duplicated().sum() == 0
    assert cleaned["student_id"].is_unique
    assert len(cleaned) == 500                            # 526 raw rows - 26 duplicates


# ---------------- formats, ranges and outliers ----------------
def test_labels_are_standardised(project):
    cleaned = project["cleaned"]
    assert set(cleaned["gender"]) == {"Male", "Female"}
    assert set(cleaned["internet_access"]) == {"Yes", "No"}
    assert set(cleaned["extra_classes"]) == {"Yes", "No"}
    assert set(cleaned["school_type"]) == set(config.SCHOOL_TYPES)
    assert set(cleaned["parental_education"]) == set(config.EDUCATION_ORDER)


def test_numeric_columns_are_numbers_inside_valid_ranges(project):
    cleaned = project["cleaned"]
    for column, (low, high) in config.VALID_RANGES.items():
        assert pd.api.types.is_numeric_dtype(cleaned[column]), column
        assert cleaned[column].between(low, high).all(), column


def test_outliers_are_handled(project):
    cleaned, report = project["cleaned"], project["report"]
    assert report["outliers_total"] > 0                    # there were outliers ...
    assert report["outliers_after"] == 0                   # ... and none remain
    assert cleaned["study_hours_per_week"].max() < 45      # 150 hours/week is gone


def test_exam_date_is_a_real_date(project):
    assert pd.api.types.is_datetime64_any_dtype(project["cleaned"]["exam_date"])


def test_unnecessary_columns_are_removed(project):
    for column in config.COLUMNS_TO_DROP:
        assert column not in project["cleaned"].columns


# ---------------- saved files ----------------
def test_cleaned_csv_is_saved_and_can_be_reloaded(project):
    assert project["cleaned_path"].exists()
    reloaded = pd.read_csv(project["cleaned_path"])
    assert reloaded.shape == project["cleaned"].shape
    assert project["report_path"].exists()


def test_load_raw_data_treats_question_mark_as_missing(project):
    raw_file = project["folder"] / "raw_data.csv"
    assert "?" not in load_raw_data(raw_file)["math_score"].astype(str).values


def test_comparison_table_has_before_and_after(project):
    table = build_comparison_table(project["report"])
    assert list(table.columns) == ["Measure", "Before cleaning", "After cleaning"]
    assert (table["After cleaning"] <= table["Before cleaning"]).all()


# ---------------- analysis ----------------
def test_analysis_tables_and_observations(project, tmp_path):
    cleaned = project["cleaned"]
    assert descriptive_statistics(cleaned).shape[0] == len(config.ANALYSIS_COLUMNS)
    assert {"mean", "median", "mode"} <= set(central_tendency(cleaned).columns)
    corr = correlation_matrix(cleaned)
    assert corr.shape[0] == corr.shape[1] and (corr.abs() <= 1).all().all()
    assert len(generate_observations(cleaned)) >= 8
    run_analysis(cleaned, output_dir=tmp_path, log=lambda *_: None)
    assert (tmp_path / "analysis_report.txt").exists()


# ---------------- graphs ----------------
def test_all_graphs_are_created(project, tmp_path):
    saved = create_all_graphs(project["cleaned"], project["raw"], output_dir=tmp_path / "graphs",
                              log=lambda *_: None)
    assert len(saved) == 7
    for path in saved.values():
        assert path.exists() and path.stat().st_size > 5000     # a real, non-empty picture
