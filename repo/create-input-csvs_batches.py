import pandas as pd
import os

# Define input and output folders
input_folder = '/home/owusus/Documents/GitHub/nsanku/repo/parallel/verses'  # <-- replace with your input folder
output_folder = '/home/owusus/Documents/GitHub/nsanku/input'  # <-- replace with your desired output folder

# Set the number of rows per CSV in each batch
rows_per_csv = 1000  # <-- Each CSV in each batch will have exactly 200 rows

# Ensure the output folder exists, create it if it doesn't
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize summary data
summary_data = []

# First, read all CSV files and get their row counts
csv_files = {}
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        input_file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(input_file_path)
        csv_files[filename] = {
            'df': df,
            'total_rows': len(df),
            'processed_rows': 0
        }
        summary_data.append({
            'File': filename,
            'Total Rows': len(df),
            'Batches Created': 0
        })

# Determine how many complete batches we can create
if not csv_files:
    print("No CSV files found in the input folder!")
    exit()

# Find the maximum number of complete batches we can create
max_batches = min([df_info['total_rows'] // rows_per_csv for df_info in csv_files.values()])
print(f"Can create {max_batches} complete batches")

# Create batches
for batch_num in range(1, max_batches + 1):
    # Create batch folder
    batch_folder = os.path.join(output_folder, f'batch-{batch_num}')
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)
    
    print(f"Creating batch {batch_num}...")
    
    # Process each CSV file for this batch
    for filename, df_info in csv_files.items():
        # Calculate start and end indices for this batch
        start_idx = df_info['processed_rows']
        end_idx = start_idx + rows_per_csv
        
        # Get the batch data
        batch_df = df_info['df'].iloc[start_idx:end_idx].copy()
        
        # Rename columns
        batch_df = batch_df.rename(columns={
            'translation_verse': 'text',
            'english_verse': 'ref'
        })
        
        # Drop the 'verse_number' column if it exists
        if 'verse_number' in batch_df.columns:
            batch_df = batch_df.drop(columns=['verse_number'])
        
        # Create output filename
        base_name = filename[:-4]  # Remove the .csv extension
        output_filename = f"{base_name}-eng.csv"
        output_file_path = os.path.join(batch_folder, output_filename)
        
        # Save the batch CSV
        batch_df.to_csv(output_file_path, index=False)
        
        # Update processed rows count
        df_info['processed_rows'] = end_idx
        
        # Update summary data
        for item in summary_data:
            if item['File'] == filename:
                item['Batches Created'] = batch_num

# Update summary with final batch counts
for item in summary_data:
    filename = item['File']
    item['Rows Processed'] = csv_files[filename]['processed_rows']
    item['Rows Remaining'] = csv_files[filename]['total_rows'] - csv_files[filename]['processed_rows']

# Create summary dataframe
summary_df = pd.DataFrame(summary_data)

# Save summary to text file
summary_txt_path = os.path.join(output_folder, 'processing_summary.txt')
with open(summary_txt_path, 'w') as f:
    f.write("📊 Processing Summary:\n")
    f.write("=" * 80 + "\n")
    f.write("SOURCE FILES AND BATCH INFORMATION:\n")
    f.write("=" * 80 + "\n")
    f.write(summary_df.to_string(index=False) + "\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("BATCH FOLDERS CREATED:\n")
    f.write("=" * 80 + "\n")
    for batch_num in range(1, max_batches + 1):
        batch_folder = os.path.join(output_folder, f'batch-{batch_num}')
        num_files = len([f for f in os.listdir(batch_folder) if f.endswith('.csv')])
        f.write(f"batch-{batch_num}: {num_files} CSV files\n")
    
    f.write("=" * 80 + "\n")
    f.write(f"Total Batches Created: {max_batches}\n")
    f.write(f"Total Rows Processed: {sum([item['Rows Processed'] for item in summary_data])}\n")
    f.write(f"Total Source Files: {len(summary_data)}\n")
    f.write("=" * 80 + "\n")
    f.write("All files processed and batched successfully! ✨\n")

# Display summary
print("\n📊 Processing Summary:")
print("=" * 80)
print("SOURCE FILES AND BATCH INFORMATION:")
print("=" * 80)
print(summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("BATCH FOLDERS CREATED:")
print("=" * 80)
for batch_num in range(1, max_batches + 1):
    batch_folder = os.path.join(output_folder, f'batch-{batch_num}')
    num_files = len([f for f in os.listdir(batch_folder) if f.endswith('.csv')])
    print(f"batch-{batch_num}: {num_files} CSV files")

# Print total statistics
print("=" * 80)
print(f"Total Batches Created: {max_batches}")
print(f"Total Rows Processed: {sum([item['Rows Processed'] for item in summary_data])}")
print(f"Total Source Files: {len(summary_data)}")
print("=" * 80)
print(f"Summary saved to: {summary_txt_path}")
print("All files processed and batched successfully! ✨")
