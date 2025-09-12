import json
import pandas as pd
import os
import argparse  # NEW: Import argparse


def convert_jsonl_to_csv(jsonl_path, csv_path):
    """
    Reads a .jsonl file with video-level annotations and converts it to a
    .csv file with one row per timestamp.
    """
    print(f"🚀 Starting conversion...")
    print(f"Reading from: {jsonl_path}")

    csv_rows = []
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                video_id = data.get("video") or data.get("video_id") or data.get("vid")
                query = data.get("query")
                timestamps = data.get("timestamps", [])

                if not all([video_id, query, timestamps]):
                    print(f"⚠️  Skipping invalid line: {line.strip()}")
                    continue

                for ts in timestamps:
                    start_frame, end_frame = ts
                    csv_rows.append({
                        "video_id": video_id,
                        "query": query,
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                    })
    except FileNotFoundError:
        print(f"❌ ERROR: The input file was not found at '{jsonl_path}'. Please check the path.")
        return
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred: {e}")
        return

    if not csv_rows:
        print("🤷 No data was converted. The output file will not be created.")
        return

    df = pd.DataFrame(csv_rows)
    output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"✅ Conversion complete!")
    print(f"Successfully created {len(df)} entries.")
    print(f"Output saved to: {csv_path}")


if __name__ == "__main__":
    # --- NEW: Use argparse to get file paths from the command line ---
    parser = argparse.ArgumentParser(description="Convert a .jsonl annotation file to a .csv triplet file.")
    parser.add_argument("--input-jsonl", required=True, help="Path to the input .jsonl file.")
    parser.add_argument("--output-csv", required=True, help="Path to save the output .csv file.")
    args = parser.parse_args()

    convert_jsonl_to_csv(args.input_jsonl, args.output_csv)