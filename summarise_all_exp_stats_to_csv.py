import pandas as pd
from pathlib import Path
import ast
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch analysis on multiple results.")
    parser.add_argument("--save_base", default="/group/jug/aman/experiment_analysis_results")
    return parser.parse_args()


def summarize_all_experiments(base_dir, output_excel="summary_results.xlsx"):
    """
    Traverse experiment analysis results and compile OG/WIN stats into one Excel file.
    Adds a second sheet with averages across channels.
    """
    base_dir = Path(base_dir)
    summary_rows = []
    avg_rows = []

    for csv_path in base_dir.rglob("stats_og_*.csv"):
        dataset_name = csv_path.stem.replace("stats_og_", "")
        lc_dir = csv_path.parent

        win_path = lc_dir / f"stats_win_{dataset_name}.csv"
        if not win_path.exists():
            print(f"⚠️ Missing WIN file for: {csv_path}")
            continue

        df_og = pd.read_csv(csv_path, index_col=0)
        df_win = pd.read_csv(win_path, index_col=0)

        # Extract identifiers
        try:
            dataset = csv_path.parts[-4]
            modality = csv_path.parts[-3]
            lc_type = csv_path.parts[-2]
        except IndexError:
            dataset, modality, lc_type = "Unknown", "Unknown", "Unknown"

        def parse_tuple_string(s):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, list) and len(val) == 2 and isinstance(val[0], tuple):
                    return [x[0] for x in val]  # mean values only
                return val
            except Exception:
                return [None, None]

        og_vals = df_og.iloc[0].apply(parse_tuple_string)
        win_vals = df_win.iloc[0].apply(parse_tuple_string)

        # --- Per-channel summary ---
        record = {
            "Dataset": dataset,
            "Modality": modality,
            "LC_Type": lc_type,
        }

        # --- Average across channels summary ---
        avg_record = {
            "Dataset": dataset,
            "Modality": modality,
            "LC_Type": lc_type,
        }

        for col in df_og.columns:
            ch0_og, ch1_og = og_vals[col]
            ch0_win, ch1_win = win_vals[col]

            record[f"{col}_Ch0_OG"] = ch0_og
            record[f"{col}_Ch1_OG"] = ch1_og
            record[f"{col}_Ch0_WIN"] = ch0_win
            record[f"{col}_Ch1_WIN"] = ch1_win

            # Compute average across channels
            avg_og = None if ch0_og is None or ch1_og is None else (ch0_og + ch1_og) / 2
            avg_win = None if ch0_win is None or ch1_win is None else (ch0_win + ch1_win) / 2

            avg_record[f"{col}_Avg_OG"] = avg_og
            avg_record[f"{col}_Avg_WIN"] = avg_win

        summary_rows.append(record)
        avg_rows.append(avg_record)

    # Combine into DataFrames
    summary_df = pd.DataFrame(summary_rows).sort_values(by=["Dataset", "Modality", "LC_Type"], ignore_index=True)
    avg_df = pd.DataFrame(avg_rows).sort_values(by=["Dataset", "Modality", "LC_Type"], ignore_index=True)

    # Save to Excel with two sheets
    output_path = Path(base_dir) / output_excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name="Per_Channel", index=False)
        avg_df.to_excel(writer, sheet_name="Channel_Averages", index=False)

    print(f"✅ Summary saved to: {output_path}")
    return summary_df, avg_df


if __name__ == "__main__":
    # args = parse_args()
    summary_df, avg_df = summarize_all_experiments(
        base_dir="./local_temp_dir",
        output_excel="consolidated_experiments_summary.xlsx"
    )
