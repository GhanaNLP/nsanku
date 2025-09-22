import os
import re
import pandas as pd

def extract_chapter(url):
    """Extract chapter reference like GEN.1 from URL."""
    match = re.search(r'/bible/\d+/([1-3]?[A-Z]{2,}\.\d+)(?:\.|$)', url)
    return match.group(1) if match else None

# Paths
english_file = 'bible_chapters_output-eng.csv'
root_folder = 'langs-data'  # folder containing language subfolders
output_root = 'parallel'
os.makedirs(output_root, exist_ok=True)

# Load English CSV
eng_df = pd.read_csv(english_file)
eng_df['chapter'] = eng_df['URL'].apply(extract_chapter)

# Loop through language subfolders
for lang_folder in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, lang_folder)
    if os.path.isdir(folder_path):
        combined_aligned = []

        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                lang_path = os.path.join(folder_path, file)
                lang_df = pd.read_csv(lang_path)
                lang_df['chapter'] = lang_df['URL'].apply(extract_chapter)

                # Merge English and target language on chapter
                merged_df = pd.merge(
                    eng_df,
                    lang_df,
                    on='chapter',
                    suffixes=('_english', '_translation')
                )

                # Keep only relevant columns
                merged_df = merged_df[['chapter', 'URL_english', 'Content_english', 'URL_translation', 'Content_translation']]
                merged_df.rename(columns={
                    'URL_english': 'english_url',
                    'Content_english': 'english_text',
                    'URL_translation': 'translation_url',
                    'Content_translation': 'translation_text'
                }, inplace=True)

                combined_aligned.append(merged_df)

        # Combine all CSVs for this language folder
        if combined_aligned:
            final_df = pd.concat(combined_aligned, ignore_index=True)
            output_path = os.path.join(output_root, f"{lang_folder}.csv")
            final_df.to_csv(output_path, index=False)
            print(f"Saved {len(final_df)} matched rows for {lang_folder} at {output_path}")

