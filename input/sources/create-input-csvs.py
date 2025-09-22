import pandas as pd
import os

# Define input and output folders
input_folder = '/home/owusus/Documents/GitHub/nsanku/input/sources/parallel/verses'  # <-- replace with your input folder
output_folder = '/home/owusus/Documents/GitHub/nsanku/input'  # <-- replace with your desired output folder

# Ensure the output folder exists, create it if it doesn't
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize summary data
summary_data = []

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        # Construct the full path for the input file
        input_file_path = os.path.join(input_folder, filename)
        
        # Read the CSV
        df = pd.read_csv(input_file_path)
        
        # Rename columns
        df = df.rename(columns={
            'translation_verse': 'text',
            'english_verse': 'ref'
        })
        
        # Drop the 'verse_number' column if it exists
        if 'verse_number' in df.columns:
            df = df.drop(columns=['verse_number'])
            
        # Create new filename with -eng appended before the .csv extension
        base_name = filename[:-4]  # Remove the .csv extension
        new_filename = f"{base_name}-eng.csv"
        
        # Construct the full path for the output file
        output_file_path = os.path.join(output_folder, new_filename)
        
        # Save the modified CSV to the new output folder
        df.to_csv(output_file_path, index=False)
        
        # Record the final row count
        final_rows = len(df)
        
        # Add to summary data (only processed filename)
        summary_data.append({
            'Processed File': new_filename,
            'Rows': final_rows
        })

# Create summary dataframe and sort by row count (descending)
summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Rows', ascending=False)

# Save summary to text file
summary_txt_path = os.path.join(output_folder, 'processing_summary.txt')
with open(summary_txt_path, 'w') as f:
    f.write("📊 Processing Summary:\n")
    f.write("=" * 50 + "\n")
    f.write(summary_df.to_string(index=False) + "\n")
    f.write("=" * 50 + "\n")
    f.write(f"Total Files Processed: {len(summary_data)}\n")
    f.write(f"Total Rows Processed: {summary_df['Rows'].sum():,}\n")
    f.write("=" * 50 + "\n")
    f.write("All CSV files processed and saved successfully! ✨\n")

# Display summary
print("\n📊 Processing Summary (Sorted by Row Count - Highest to Lowest):")
print("=" * 50)
print(summary_df.to_string(index=False))

# Print total statistics
print("=" * 50)
print(f"Total Files Processed: {len(summary_data)}")
print(f"Total Rows Processed: {summary_df['Rows'].sum():,}")
print("=" * 50)
print(f"Summary saved to: {summary_txt_path}")
print("All CSV files processed and saved successfully! ✨")
