"""
main.py  -  Run the complete project from the command line
==========================================================
    python main.py                 # run everything
    python main.py --regenerate    # create a fresh raw dataset first

What it does (in order):
    0. Creates dataset/raw_data.csv if it does not exist (synthetic data)
    1. Cleans the raw data         -> dataset/cleaned_data.csv
    2. Runs the data analysis      -> outputs/analysis_report.txt (+ CSV tables)
    3. Creates all the charts      -> outputs/graphs/*.png
    4. Saves everything printed    -> outputs/run_log.txt

Afterwards start the dashboard with:   streamlit run app.py
"""

import argparse

from src import config
from src.data_analysis import run_analysis
from src.data_cleaning import run_cleaning
from src.data_visualization import create_all_graphs
from src.generate_dataset import generate_raw_dataset

LOG_LINES = []


def log(message=""):
    """Print a message and remember it for outputs/run_log.txt."""
    print(message)
    LOG_LINES.append(str(message))


def main():
    parser = argparse.ArgumentParser(description="Data Cleaning & Visualization project")
    parser.add_argument("--regenerate", action="store_true",
                        help="create a new synthetic raw dataset before cleaning")
    args = parser.parse_args()

    log("#" * 72)
    log("#  DATA CLEANING & VISUALIZATION PROJECT")
    log("#" * 72)

    # ---- Step 0: raw dataset ----
    if args.regenerate or not config.RAW_PATH.exists():
        generate_raw_dataset()
        log(f"Created the synthetic raw dataset: {config.RAW_PATH}")
    else:
        log(f"Using the existing raw dataset: {config.RAW_PATH}")

    # ---- Step 1: cleaning ----
    raw_df, cleaned_df, _ = run_cleaning(log=log)

    # ---- Step 2: analysis ----
    run_analysis(cleaned_df, log=log)

    # ---- Step 3: charts ----
    log("")
    log("=" * 72)
    log("DATA VISUALIZATION: saving charts")
    log("=" * 72)
    create_all_graphs(cleaned_df, raw_df, log=log)
    log(f"  saved {config.OUTPUT_DIR / 'graph_explanations.md'}")

    # ---- Finished ----
    log("")
    log("=" * 72)
    log("ALL DONE")
    log("=" * 72)
    log(f"Cleaned dataset : {config.CLEANED_PATH}")
    log(f"Graphs          : {config.GRAPH_DIR}")
    log("Next step       : streamlit run app.py")

    (config.OUTPUT_DIR / "run_log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
