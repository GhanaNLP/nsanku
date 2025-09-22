#!/usr/bin/env python3
"""
scrape.py  –  batch-click SVG items and collect overlay URLs
Input : input.csv  (lang,url)
Output: output.csv (original_url,lang,grabbed_url)
"""
import csv, time, sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException

INPUT_FILE  = "doc-page-urls.csv"
OUTPUT_FILE = "docs-urls.csv"

# -----------------------------------------------------------
# Fire-up Firefox (geckodriver on PATH)
# -----------------------------------------------------------
service = Service()
options = Options()
options.binary_location = "/snap/firefox/current/usr/lib/firefox/firefox"
driver = webdriver.Firefox(service=service, options=options)
wait = WebDriverWait(driver, 30)  # Increased timeout
short_wait = WebDriverWait(driver, 5)

# -----------------------------------------------------------
# XPath patterns to try (in order of preference)
# -----------------------------------------------------------
CLICKABLE_ELEMENT_XPATHS = [
    # Original SVG patterns
    "//div[6]/div/div/main/div/div/div[2]/div/div[2]/div[1]/div[3]/a/span[1]/svg",
    "//div[6]/div/div/main/div/div/div[2]/div/div[2]/div[1]/div[3]/a/span[1]",
    
    # New patterns from your examples
    "//div[6]/div/div/main/div/div[1]/div[4]/div/div[2]/div/div[2]/div[1]/div[3]/a/span[1]",
    
    # CSS selector converted to XPath
    "//div[contains(@class, 'pub-wp')]//div[contains(@class, 'downloadLinks')]//a[contains(@class, 'jsDownload')]//span[contains(@class, 'buttonIcon')]",
    
    # More flexible patterns
    "//main//div[contains(@class, 'downloadLinks')]//a//span[1]",
    "//main//a[contains(@class, 'jsDownload')]//span",
    "//main//span[contains(@class, 'buttonIcon')]",
    
    # Fallback - any clickable download elements
    "//main//a[contains(@href, 'download') or contains(@class, 'download') or contains(@class, 'jsDownload')]"
]

OVERLAY_URL_XPATH = "/html/body/div[10]/div/div/div[2]/div/div[3]/div[3]/div[1]/a"

# Alternative overlay selectors to try
OVERLAY_URL_SELECTORS = [
    "/html/body/div[10]/div/div/div[2]/div/div[3]/div[3]/div[1]/a",
    "//div[contains(@class, 'overlay') or contains(@class, 'modal') or contains(@class, 'popup')]//a[contains(@href, 'http')]",
    "//div[@role='dialog']//a[contains(@href, 'http')]",
    "//*[contains(@class, 'download')]//a[contains(@href, 'http')]"
]

# -----------------------------------------------------------
# Helper: Wait for page to be fully loaded
# -----------------------------------------------------------
def wait_for_page_load(url):
    """Wait for page to be fully loaded with multiple checks"""
    print("Waiting for page to load...")
    
    # Wait for main content
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "main")))
        print("✓ Main content loaded")
    except TimeoutException:
        print("WARNING: Main content not found")
    
    # Wait for body to have the expected classes (indicates JS has run)
    try:
        wait.until(lambda driver: "jsPageReady" in driver.find_element(By.TAG_NAME, "body").get_attribute("class"))
        print("✓ JavaScript initialization complete")
    except TimeoutException:
        print("WARNING: JavaScript may not be fully initialized")
    
    # Additional wait for dynamic content
    time.sleep(3)
    print("✓ Page load complete")

# -----------------------------------------------------------
# Helper: Find clickable elements using multiple strategies
# -----------------------------------------------------------
def find_clickable_elements():
    """Try multiple XPath patterns to find clickable elements"""
    elements = []
    
    for i, xpath in enumerate(CLICKABLE_ELEMENT_XPATHS):
        try:
            print(f"Trying pattern {i+1}: {xpath}")
            found_elements = driver.find_elements(By.XPATH, xpath)
            if found_elements:
                print(f"✓ Found {len(found_elements)} elements with pattern {i+1}")
                elements = found_elements
                break
            else:
                print(f"✗ No elements found with pattern {i+1}")
        except Exception as e:
            print(f"✗ Error with pattern {i+1}: {e}")
            continue
    
    if not elements:
        print("Trying fallback: looking for any download-related links...")
        try:
            elements = driver.find_elements(By.XPATH, "//a[contains(text(), 'download') or contains(@title, 'download')]")
            if elements:
                print(f"✓ Found {len(elements)} elements with fallback pattern")
        except:
            pass
    
    return elements

# -----------------------------------------------------------
# Helper: Get overlay URL using multiple selectors
# -----------------------------------------------------------
def get_overlay_url():
    """Try multiple selectors to find the overlay URL"""
    for i, selector in enumerate(OVERLAY_URL_SELECTORS):
        try:
            print(f"Looking for overlay with selector {i+1}...")
            overlay_element = wait.until(EC.presence_of_element_located((By.XPATH, selector)))
            url = overlay_element.get_attribute("href")
            if url:
                print(f"✓ Found overlay URL: {url}")
                return url
        except TimeoutException:
            print(f"✗ Overlay selector {i+1} timed out")
            continue
        except Exception as e:
            print(f"✗ Error with overlay selector {i+1}: {e}")
            continue
    
    return None

# -----------------------------------------------------------
# Helper: Close overlay/modal
# -----------------------------------------------------------
def close_overlay():
    """Try multiple methods to close the overlay"""
    methods = [
        lambda: driver.execute_script("document.dispatchEvent(new KeyboardEvent('keydown', {'key': 'Escape'}));"),
        lambda: driver.execute_script("window.dispatchEvent(new KeyboardEvent('keydown', {'key': 'Escape'}));"),
        lambda: driver.find_element(By.XPATH, "//button[contains(@class, 'close') or contains(@aria-label, 'close')]").click(),
        lambda: driver.find_element(By.XPATH, "//*[@data-dismiss='modal' or @data-bs-dismiss='modal']").click()
    ]
    
    for i, method in enumerate(methods):
        try:
            method()
            time.sleep(0.5)
            print(f"✓ Overlay closed using method {i+1}")
            return
        except:
            continue
    
    print("WARNING: Could not close overlay")

# -----------------------------------------------------------
# Helper: process one page
# -----------------------------------------------------------
def process_page(lang: str, original_url: str):
    print(f"\n{'='*80}")
    print(f"Processing: {lang} - {original_url}")
    print(f"{'='*80}")
    
    try:
        driver.get(original_url)
        wait_for_page_load(original_url)
    except Exception as e:
        print(f"ERROR: Failed to load page: {e}")
        return []
    
    # Find all clickable elements
    clickable_elements = find_clickable_elements()
    
    if not clickable_elements:
        print("ERROR: No clickable elements found on this page")
        return []
    
    print(f"Found {len(clickable_elements)} clickable elements to process")
    
    results = []
    for idx, element in enumerate(clickable_elements, start=1):
        print(f"\n--- Processing element {idx}/{len(clickable_elements)} ---")
        
        try:
            # Scroll element into view
            driver.execute_script("arguments[0].scrollIntoView({block:'center', behavior: 'smooth'});", element)
            time.sleep(1)
            
            # Wait for element to be clickable
            try:
                WebDriverWait(driver, 5).until(EC.element_to_be_clickable(element))
            except TimeoutException:
                print("WARNING: Element not clickable, trying anyway...")
            
            # Try multiple click methods
            click_success = False
            click_methods = [
                lambda: element.click(),
                lambda: driver.execute_script("arguments[0].click();", element),
                lambda: element.find_element(By.XPATH, "./ancestor-or-self::a[1]").click()
            ]
            
            for method_idx, click_method in enumerate(click_methods):
                try:
                    click_method()
                    click_success = True
                    print(f"✓ Clicked using method {method_idx + 1}")
                    break
                except Exception as e:
                    print(f"✗ Click method {method_idx + 1} failed: {e}")
                    continue
            
            if not click_success:
                print("ERROR: All click methods failed")
                continue
            
            # Wait a bit for overlay to appear
            time.sleep(2)
            
            # Try to get the overlay URL
            grabbed_url = get_overlay_url()
            
            if grabbed_url:
                results.append(grabbed_url)
                print(f"✓ SUCCESS: Got URL for element {idx}")
            else:
                print(f"✗ FAILED: No URL found for element {idx}")
            
            # Close overlay
            close_overlay()
            time.sleep(1)  # Wait before processing next element
            
        except Exception as e:
            print(f"ERROR processing element {idx}: {e}")
            # Try to close any open overlay before continuing
            close_overlay()
            continue
    
    print(f"\n✓ Page processing complete. Found {len(results)} URLs.")
    return results

# -----------------------------------------------------------
# Main loop
# -----------------------------------------------------------
def main():
    print("Starting batch download URL scraper...")
    
    # Read input file
    try:
        with open(INPUT_FILE, newline='', encoding='utf-8') as fin:
            reader = csv.DictReader(fin)
            if 'lang' not in reader.fieldnames or 'url' not in reader.fieldnames:
                sys.exit("ERROR: input.csv must contain headers: lang,url")
            rows = list(reader)
            print(f"✓ Loaded {len(rows)} URLs to process")
    except FileNotFoundError:
        sys.exit(f"ERROR: Input file {INPUT_FILE} not found")
    except Exception as e:
        sys.exit(f"ERROR reading input file: {e}")
    
    # Process URLs and write results
    total_urls_found = 0
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(["original_url", "lang", "grabbed_url"])
        
        for row_num, row in enumerate(rows, start=1):
            lang = row['lang'].strip()
            url = row['url'].strip()
            
            if not url:
                print(f"WARNING: Empty URL in row {row_num}, skipping")
                continue
            
            print(f"\n🔄 Processing {row_num}/{len(rows)}")
            
            try:
                grabbed_urls = process_page(lang, url)
                
                if grabbed_urls:
                    for gurl in grabbed_urls:
                        writer.writerow([url, lang, gurl])
                    fout.flush()  # Ensure data is written immediately
                    total_urls_found += len(grabbed_urls)
                    print(f"✅ SUCCESS: {len(grabbed_urls)} URLs saved for {lang}")
                else:
                    print(f"❌ FAILED: No URLs found for {lang} - {url}")
                    
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
                break
            except Exception as e:
                print(f"💥 CRITICAL ERROR processing {url}: {e}", file=sys.stderr)
                continue
    
    # Cleanup
    driver.quit()
    
    print(f"\n{'='*80}")
    print(f"🎉 BATCH PROCESSING COMPLETE!")
    print(f"📊 Total URLs found: {total_urls_found}")
    print(f"📁 Results saved to: {OUTPUT_FILE}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()