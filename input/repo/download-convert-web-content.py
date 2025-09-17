import os
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

# --- SETTINGS ---
xml_file = 'kea/sitemap.xml'   # Replace with your actual XML filename
base_output_folder = 'kea/downloaded_pages'

# Create subfolder for HTML files
html_folder = os.path.join(base_output_folder, 'html_files')

# Ensure the folder exists
os.makedirs(html_folder, exist_ok=True)

# Parse the XML
tree = ET.parse(xml_file)
root = tree.getroot()

# XML has a namespace, so we need to include it
namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

# Extract URLs from <loc> tags
urls = [elem.text for elem in root.findall('.//ns:loc', namespaces=namespace)]

# Function to download a single page
def download_page(url, index):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Save HTML file in the html_folder
        html_filename = os.path.join(html_folder, f'page_{index}.html')
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f'Downloaded HTML: {url}')
    except Exception as e:
        print(f'Failed to download {url}: {e}')

# Use ThreadPoolExecutor to download pages in parallel
with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers to control the number of concurrent downloads
    executor.map(lambda i: download_page(urls[i], i + 1), range(len(urls)))

# Now, batch convert all HTML files to text using Calibre
import subprocess

# Create a subfolder for text files
text_folder = os.path.join(base_output_folder, 'text_files')
os.makedirs(text_folder, exist_ok=True)

# Convert each HTML file to text
for i, url in enumerate(urls, start=1):
    html_filename = os.path.join(html_folder, f'page_{i}.html')
    text_filename = os.path.join(text_folder, f'page_{i}.txt')
    
    # Convert using Calibre's ebook-convert
    ebook_convert_command = ['ebook-convert', html_filename, text_filename]
    
    # Run the conversion command
    subprocess.run(ebook_convert_command, check=True)

    print(f'Converted to text: {text_filename}')
