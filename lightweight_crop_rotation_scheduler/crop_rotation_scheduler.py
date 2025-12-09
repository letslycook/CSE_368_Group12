#!/usr/bin/env python3
import pandas as pd
import subprocess
import json
import re
import argparse

MODEL_NAME = "llama3:8b"   # or another local model you prefer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight crop rotation scheduler using Ollama."
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input CSV or Excel file containing field history."
    )
    
    parser.add_argument(
        "-off",
        type=int,
        default=5,
        help="OFF-year interval (every N years, a plot must go OFF). Default: 5"
    )
    parser.add_argument(
        "-poly",
        type=str,
        default="T",
        help="Enable or disable polyculture: T/True or F/False (case-insensitive). Default: T"
    )
    return parser.parse_args()


def parse_poly_flag(poly_str: str) -> bool:
    """
    Convert -poly argument into a boolean.
    Accepts: T, True, F, False (case-insensitive), also Y/N, 1/0 for robustness.
    """
    if poly_str is None:
        return True
    s = poly_str.strip().lower()
    if s in ("t", "true", "y", "yes", "1"):
        return True
    if s in ("f", "false", "n", "no", "0"):
        return False
    # If weird value, default to True but warn
    print(f"Unrecognized -poly value '{poly_str}', defaulting to True.")
    return True


def load_input_file(path):
    """
    Load the standardized input crop rotation CSV/Excel file
    and strip any stray whitespace from column names.
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # Clean up stray whitespace in column names
    df.columns = df.columns.str.strip()

    return df


def build_prompt(df, off_interval, use_polyculture=True):
    """
    Build the prompt sent to Ollama based on the rules & field history.
    """

    # Convert dataframe into a list of plot records for the LLM
    history_records = []
    for _, row in df.iterrows():
        history_records.append({
            "Plot_ID": int(row["Plot_ID"]),
            "Three_Seasons_Ago": row["3_Seasons_Ago"],
            "Two_Seasons_Ago": row["2_Seasons_Ago"],
            "Last_Season": row["Last_Season"],
            "Years_Since_OFF": int(row["Years_Since_OFF-Year"])
        })

    history_json = json.dumps(history_records, indent=2)

    base_prompt = f"""
You are a crop rotation planning assistant. You will receive field data for 15 plots and must assign
a crop (or OFF) for the next season using the following rules.

===========================
ROTATION CONSTRAINTS
===========================

1. Most important rule:
   No crop may be planted in the same plot two years in a row.

2. OFF-year rule (user-defined interval):
   Every plot must have an OFF year every {off_interval} years.
   If some plots are overdue while others are not, plots with the largest "Years_Since_OFF" are
   highest priority for OFF.

3. Optional polyculture rule:
   If polyculture is enabled, avoid placing the same crop type directly adjacent
   (up, down, left, right) to another of its kind.
   (Assume plots are arranged in a 3×5 grid with IDs 1–15 left-to-right, top-to-bottom.)

Polyculture status: {"ENABLED" if use_polyculture else "DISABLED"}

===========================
FIELD HISTORY INPUT
===========================
{history_json}

===========================
EXPECTED OUTPUT FORMAT
===========================
Return ONLY a JSON list (array) with exactly 15 objects, each structured as:

[
  {{
    "Plot_ID": X,
    "Current_Season_Suggested": "crop_name_or_OFF"
  }},
  ...
]

IMPORTANT:
- Do NOT include any explanation, markdown, or text before or after the JSON.
- Do NOT wrap it in backticks.
- Output ONLY the raw JSON array.
"""

    return base_prompt


def call_ollama(prompt, model=MODEL_NAME):
    """
    Call Ollama via subprocess and return the model's JSON output.
    We robustly extract the JSON array from the text the model returns.
    """
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True
    )

    raw = result.stdout.strip()

    # Try to extract the JSON array between the first '[' and last ']'
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        print("ERROR: Could not find a JSON array in Ollama output.")
        print("Raw output:\n", raw)
        raise ValueError("No JSON array found in Ollama output.")

    json_str = match.group(0)

    try:
        output_json = json.loads(json_str)
        return output_json
    except json.JSONDecodeError:
        print("ERROR: Ollama JSON still invalid.")
        print("Extracted JSON string:\n", json_str)
        raise


def convert_llm_output_to_dataframe(output_json, df_history, off_interval):
    """
    Convert Ollama's structured JSON into a final output DataFrame.

    Logic:
    - Use Ollama's suggested crop for each plot.
    - Choose ONE plot to be OFF:
        * The plot(s) with the largest Years_Since_OFF-Year are highest priority.
        * Among them, pick the lowest Plot_ID (tie-breaker).
    - For that OFF plot: Years_Since_OFF-Year = 0.
    - For all others: Years_Since_OFF-Year = previous + 1 (if not OFF).
    """

    # Map Plot_ID -> previous Years_Since_OFF-Year
    history_years = {
        int(row["Plot_ID"]): int(row["Years_Since_OFF-Year"])
        for _, row in df_history.iterrows()
    }

    # Optional future use: last-season crop
    last_season_crop = {
        int(row["Plot_ID"]): str(row["Last_Season"]).strip()
        for _, row in df_history.iterrows()
    }

    # Map Plot_ID -> LLM-suggested crop (sanitized)
    llm_crops = {}
    for item in output_json:
        pid = int(item["Plot_ID"])
        crop = str(item["Current_Season_Suggested"]).strip()
        llm_crops[pid] = crop

    # Decide which plot becomes OFF this season
    # Priority: largest Years_Since_OFF-Year
    max_years = max(history_years.values())
    off_plot_id = None

    if max_years >= off_interval:
        # candidates with the maximum years since OFF
        candidates = [pid for pid, yrs in history_years.items() if yrs == max_years]
        off_plot_id = min(candidates)  # simple tie-breaker: smallest Plot_ID

    # Build final rows, recomputing Years_Since_OFF-Year deterministically
    rows = []
    for pid in sorted(history_years.keys()):
        prev_years = history_years[pid]
        crop = llm_crops.get(pid, "OFF")

        # Force OFF for the selected plot
        if off_plot_id is not None and pid == off_plot_id:
            crop = "OFF"
            new_years = 0
        else:
            # If not OFF: increment years since OFF by 1
            if crop.upper() == "OFF":
                new_years = 0
            else:
                new_years = prev_years + 1

        rows.append({
            "Plot_ID": pid,
            "Current_Season_Suggested": crop,
            "Years_Since_OFF-Year": new_years
        })

    return pd.DataFrame(rows, columns=[
        "Plot_ID",
        "Current_Season_Suggested",
        "Years_Since_OFF-Year"
    ])


def main():
    args = parse_args()
    off_interval = args.off
    enable_polyculture = parse_poly_flag(args.poly)

    # use the -i/--input argument instead of hard-coding the path
    input_path = args.input

    print("Loading input file...")
    df = load_input_file(input_path)
    print("Loaded rows:", len(df))

    print(f"Using OFF interval: {off_interval}")
    print(f"Polyculture enabled: {enable_polyculture}")

    print("Building prompt...")
    prompt = build_prompt(df, off_interval, enable_polyculture)
    print("Prompt length (chars):", len(prompt))

    print("Calling Ollama... this may take a bit.")
    response_json = call_ollama(prompt)
    print("Got response from Ollama.")

    final_df = convert_llm_output_to_dataframe(response_json, df, off_interval)

    print("\n===== FINAL SCHEDULE =====\n")
    print(final_df.to_string(index=False))

    # Save output file
    final_df.to_csv("OUTPUT_Crop_Rotation_Field_Map.csv", index=False)
    print("\nSaved as OUTPUT_Crop_Rotation_Field_Map.csv\n")


if __name__ == "__main__":
    main()