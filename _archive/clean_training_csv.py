import csv
import json
import os
import shutil

from app import _clean_trainer_review_text

CSV_PATH = "review_training_labels.csv"
BAK_PATH = "review_training_labels.bak.csv"

def clean_csv():
    if not os.path.exists(CSV_PATH):
        print(f"{CSV_PATH} not found.")
        return

    # Backup the file
    print(f"Backing up to {BAK_PATH}...")
    shutil.copyfile(CSV_PATH, BAK_PATH)

    cleaned_rows = []
    with open(BAK_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rj = row.get("reviews_json") or "[]"
            try:
                reviews = json.loads(rj)
            except Exception:
                reviews = []

            if isinstance(reviews, list):
                clean_revs = [_clean_trainer_review_text(str(r)) for r in reviews]
                # Filter out exactly empty reviews that were just owner replies
                clean_revs = [cr for cr in clean_revs if cr]
                row["reviews_json"] = json.dumps(clean_revs, ensure_ascii=False)
            
            cleaned_rows.append(row)

    print(f"Writing {len(cleaned_rows)} cleaned rows to {CSV_PATH}...")
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    print("Done!")

if __name__ == "__main__":
    clean_csv()
