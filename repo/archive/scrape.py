# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
scrape_bible_links.py  –  Extract Bible links from a list of URLs
Input : input.csv  (url)
Output: output.csv (url,bible_link)
"""
import csv
import time
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options

INPUT_FILE  = "Youversion-Ghana.csv"
OUTPUT_FILE = "Youversion-Ghana_bible-links.csv"

# -----------------------------------------------------------
# Fire-up Firefox (geckodriver on PATH)
# -----------------------------------------------------------
service = Service()
options = Options()
options.binary_location = "/snap/firefox/current/usr/lib/firefox/firefox"  # Adjust if necessary
driver = webdriver.Firefox(service=service, options=options)
wait = WebDriverWait(driver, 20)

# -----------------------------------------------------------
# XPath for the Bible link
# -----------------------------------------------------------
BIBLE_LINK_XPATH = "/html/body/div/div[2]/main/div[1]/div/div[1]/div[2]/div[2]/div/a"

# -----------------------------------------------------------
# Helper: process one page
# -----------------------------------------------------------
def process_page(url: str):
    driver.get(url)
    wait.until(EC.presence_of_element_located((By.XPATH, BIBLE_LINK_XPATH)))
    bible_link_element = driver.find_element(By.XPATH, BIBLE_LINK_XPATH)
    bible_link = bible_link_element.get_attribute("href")
    return bible_link

# -----------------------------------------------------------
# Main loop
# -----------------------------------------------------------
def main():
    with open(INPUT_FILE, newline='', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as fout:

        reader = csv.DictReader(fin)
        if 'url' not in reader.fieldnames:
            sys.exit("input.csv must contain header: url")

        writer = csv.writer(fout)
        writer.writerow(["url", "bible_link"])

        for row in reader:
            url = row['url'].strip()
            print(f"\nProcessing {url}")
            try:
                bible_link = process_page(url)
                writer.writerow([url, bible_link])
                fout.flush()
            except Exception as e:
                print(f"ERROR on {url} -> {e}", file=sys.stderr)
                writer.writerow([url, ""])
                fout.flush()
                continue

    driver.quit()
    print("\nAll done – see", OUTPUT_FILE)

if __name__ == "__main__":
    main()