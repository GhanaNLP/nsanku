import os
import re
import pandas as pd
import nltk

# Make sure the punkt tokenizer is available
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def clean_text(text: str) -> str:
    """
    Remove numbers, bible references like (Luka 3:23-38), and extra spaces.
    """
    if not isinstance(text, str):
        return ""
    # Remove Bible references in parentheses e.g. (Luka 3:23-38)
    text = re.sub(r'\([^)]*\d+[^)]*\)', '', text)
    # Remove all standalone numbers
    text = re.sub(r'\d+', '', text)
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_root_folder(root_folder: str, Content_column: str = "Content"):
    """
    For each subfolder of root_folder:
      * Read all CSVs
      * Extract and clean 'Content' column
      * Tokenize into sentences
      * Save combined .txt in root folder
    """
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        all_sentences = []

        # Process all CSV files in the subfolder
        for file in os.listdir(subfolder_path):
            if file.lower().endswith(".csv"):
                csv_path = os.path.join(subfolder_path, file)
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"Skipping {csv_path}: {e}")
                    continue

                if Content_column not in df.columns:
                    print(f"Column '{Content_column}' not in {csv_path}, skipping.")
                    continue

                for raw_text in df[Content_column]:
                    cleaned = clean_text(raw_text)
                    if cleaned:
                        sentences = sent_tokenize(cleaned)
                        all_sentences.extend(sentences)

        # Save combined text file in the root folder
        if all_sentences:
            out_path = os.path.join(root_folder, f"{subfolder}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                for s in all_sentences:
                    f.write(s + "\n")
            print(f"Saved {len(all_sentences)} sentences to {out_path}")
        else:
            print(f"No sentences found for {subfolder}")

# ==== USAGE ====
# Replace with the path to the root folder containing the subfolders
# Each subfolder should contain CSVs with a 'Content' column.
process_root_folder("/home/owusus/Documents/GitHub/nsanku/input/web-data")

