import os
import re
import pandas as pd

# --- CONFIG ---
INPUT_FOLDER = "parallel/chapters"     # folder with your CSVs
OUTPUT_FOLDER = "parallel/verses"      # output folder
EN_COL = "english_text"
TR_COL = "translation_text"

# Regex patterns
VERSE_PATTERN = re.compile(r'(\d+)\s*([^\d]+?)(?=\d|\Z)', re.DOTALL)
PAREN_PATTERN = re.compile(r'\([^)]*\)')  # remove anything inside parentheses

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def strip_parentheses(text: str) -> str:
    """Remove all content inside parentheses, including the parentheses themselves."""
    return PAREN_PATTERN.sub('', text)

def split_into_verses(text: str) -> dict:
    """
    Split a block of text into {verse_number: cleaned_text}.
    Strips extra whitespace and removes parenthetical references.
    """
    verses = {}
    for num, vtext in VERSE_PATTERN.findall(text):
        no_refs = strip_parentheses(vtext)
        clean = ' '.join(no_refs.split())
        verses[num] = clean
    return verses

def process_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    all_rows = []

    for _, row in df.iterrows():
        eng_verses = split_into_verses(str(row[EN_COL]))
        tr_verses  = split_into_verses(str(row[TR_COL]))

        # Align verses strictly by verse number
        for vnum, eng_text in eng_verses.items():
            tr_text = tr_verses.get(vnum, "")
            all_rows.append({
                "verse_number": vnum,
                "english_verse": eng_text,
                "translation_verse": tr_text
            })

    out_df = pd.DataFrame(all_rows)

    # --- Filtering ---
    # 1) Both sides have >=10 characters
    mask_chars = (
        out_df["english_verse"].str.strip().str.len().ge(10) &
        out_df["translation_verse"].str.strip().str.len().ge(10)
    )

    # 2) Word-count ratio check: shorter side >= 50% of longer side
    def ratio_ok(row):
        e_words = len(row["english_verse"].split())
        t_words = len(row["translation_verse"].split())
        if e_words == 0 or t_words == 0:
            return False
        short, long = sorted((e_words, t_words))
        return short / long >= 0.5

    mask_ratio = out_df.apply(ratio_ok, axis=1)

    out_df = out_df[mask_chars & mask_ratio]

    return out_df

def main():
    for fname in os.listdir(INPUT_FOLDER):
        if fname.lower().endswith(".csv"):
            in_path = os.path.join(INPUT_FOLDER, fname)
            out_path = os.path.join(OUTPUT_FOLDER, fname)  # keep original name
            print(f"Processing {fname} -> {out_path}")
            processed = process_csv(in_path)
            processed.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
