import csv
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, NoSuchElementException

# Chrome setup with webdriver-manager
options = Options()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Use webdriver-manager to automatically handle ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
wait = WebDriverWait(driver, 15)

# Read input CSV
input_file = "Youversion-Ghana_bible-links.csv"
with open(input_file, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    input_data = list(reader)

def retry_operation(operation, max_attempts=3, delay=2):
    """Generic retry function for Selenium operations"""
    attempt = 0
    while attempt < max_attempts:
        try:
            return operation()
        except Exception as e:
            attempt += 1
            if attempt == max_attempts:
                raise e
            print(f"⚠️  Attempt {attempt} failed. Retrying in {delay} seconds...")
            time.sleep(delay)

for row in input_data:
    url = row['url']
    lang_code = row['lang_code']
    
    # Create language folder if it doesn't exist
    os.makedirs(lang_code, exist_ok=True)
    
    # Get filename from URL
    filename = url.rstrip('/').split('/')[-1] + '.csv'
    output_file = os.path.join(lang_code, filename)
    
    # Skip if file already exists (resume capability)
    if os.path.exists(output_file):
        print(f"⏩ Skipping {url} - output file already exists")
        continue
    
    # Retry page loading with error handling
    page_loaded = False
    for load_attempt in range(3):
        try:
            print(f"Loading page (attempt {load_attempt + 1})...")
            driver.get(url)
            # Wait for page to fully load
            wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
            page_loaded = True
            break
        except Exception as e:
            print(f"❌ Page load failed: {e}")
            if load_attempt == 2:
                print("⚠️  Skipping URL after 3 failed attempts")
                continue
    
    if not page_loaded:
        continue
    
    # Create CSV file and write headers
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "Content", "URL"])
        
        chapter_count = 0
        processed_urls = set()
        consecutive_failures = 0
        max_consecutive_failures = 3

        while consecutive_failures < max_consecutive_failures:
            try:
                current_url = driver.current_url
                
                # Check for duplicate URL (prevent infinite loop)
                if current_url in processed_urls:
                    print("⚠️  Already processed this URL. Moving to next.")
                    consecutive_failures += 1
                    time.sleep(2)
                    continue
                
                processed_urls.add(current_url)
                consecutive_failures = 0  # Reset on success

                # Retry element finding with error handling
                try:
                    title_elem = retry_operation(
                        lambda: wait.until(EC.presence_of_element_located(
                            (By.XPATH, "/html/body/div/div[2]/main/div[1]/div[2]/div[1]/div[1]/div[1]/h1")
                        )),
                        max_attempts=3,
                        delay=2
                    )
                    
                    content_elem = retry_operation(
                        lambda: wait.until(EC.presence_of_element_located(
                            (By.XPATH, "/html/body/div/div[2]/main/div[1]/div[2]/div[1]/div[1]/div[1]/div")
                        )),
                        max_attempts=3,
                        delay=2
                    )
                except Exception as e:
                    print(f"❌ Failed to find elements after retries: {e}")
                    consecutive_failures += 1
                    continue

                # Extract data
                title = title_elem.text
                content = content_elem.text

                # Write to CSV
                writer.writerow([title, content, current_url])
                file.flush()  # Save immediately

                # Print status
                chapter_count += 1
                print(f"\n✅ [{lang_code}] Scraped Chapter {chapter_count}")
                print("Title:", title)
                print("URL:", current_url)
                print("Content snippet:", content[:200], "...")

                # Check if next button exists before trying to click
                next_btn_xpath = "/html/body/div/div[2]/main/div[1]/div[3]/div[2]/a/div"
                
                # First check if the next button exists at all
                try:
                    driver.find_element(By.XPATH, next_btn_xpath)
                except NoSuchElementException:
                    print("❌ No next button found. End of book.")
                    break
                
                # Try clicking next button with enhanced retry
                next_clicked = False
                
                for click_attempt in range(3):
                    try:
                        # Scroll to element to ensure it's visible
                        next_button = wait.until(EC.presence_of_element_located((By.XPATH, next_btn_xpath)))
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
                        
                        # Wait for element to be clickable
                        next_button = wait.until(EC.element_to_be_clickable((By.XPATH, next_btn_xpath)))
                        
                        # Store current URL to compare after click
                        previous_url = driver.current_url
                        
                        # Click using JavaScript as a fallback
                        if click_attempt > 0:
                            driver.execute_script("arguments[0].click();", next_button)
                        else:
                            next_button.click()
                            
                        # Wait for URL to change (indicating navigation)
                        try:
                            wait.until(EC.url_changes(previous_url))
                            # Wait for page to load after navigation
                            wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
                            # Wait for content to be present
                            wait.until(EC.presence_of_element_located(
                                (By.XPATH, "/html/body/div/div[2]/main/div[1]/div[2]/div[1]/div[1]/div[1]/h1")))
                            
                            next_clicked = True
                            break
                        except TimeoutException:
                            print(f"❌ Page didn't navigate after click (attempt {click_attempt + 1})")
                            continue
                            
                    except Exception as e:
                        print(f"❌ Next button click failed (attempt {click_attempt + 1}): {e}")
                        time.sleep(2)
                
                if not next_clicked:
                    print("❌ Failed to navigate to next chapter after 3 attempts. Moving to next URL.")
                    break

            except Exception as e:
                print(f"❌ Error while scraping: {e}")
                consecutive_failures += 1
                # Try to recover by refreshing the page
                try:
                    driver.refresh()
                    wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
                    time.sleep(3)
                except:
                    print("❌ Could not recover from error.")
                    if consecutive_failures >= max_consecutive_failures:
                        print("❌ Too many consecutive failures. Moving to next URL.")
                        break

    print(f"\n✅ Finished scraping {chapter_count} chapters from {url}. Data saved to '{output_file}'.")

# Done
driver.quit()
print("\n✅ All URLs processed!")
