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

# Chrome setup with webdriver-manager
options = Options()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Use webdriver-manager to automatically handle ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
wait = WebDriverWait(driver, 15)

# Read input CSV
input_file = "Youversion-Ghana_bible-links_fante.csv"
with open(input_file, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    input_data = list(reader)

for row in input_data:
    url = row['url']
    lang_code = row['lang_code']
    
    # Create language folder if it doesn't exist
    os.makedirs(lang_code, exist_ok=True)
    
    # Get filename from URL
    filename = url.rstrip('/').split('/')[-1] + '.csv'
    output_file = os.path.join(lang_code, filename)
    
    # Start from current URL
    driver.get(url)
    
    # Create CSV file and write headers
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "Content", "URL"])
        
        chapter_count = 0
        processed_urls = set()

        while True:
            try:
                current_url = driver.current_url
                
                # Check for duplicate URL (prevent infinite loop)
                if current_url in processed_urls:
                    print("⚠️  Already processed this URL. Moving to next.")
                    break
                processed_urls.add(current_url)

                # Wait for title and content
                title_elem = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/div/div[2]/main/div[1]/div[2]/div[1]/div[1]/div[1]/h1")))
                content_elem = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/div/div[2]/main/div[1]/div[2]/div[1]/div[1]/div[1]/div")))

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

                # Try clicking next button
                try:
                    next_btn_xpath = "/html/body/div/div[2]/main/div[1]/div[3]/div[2]/a/div"
                    next_button = wait.until(EC.element_to_be_clickable((By.XPATH, next_btn_xpath)))
                    next_button.click()
                    time.sleep(3)
                except Exception as e:
                    print("🔁 Retry clicking next button...")
                    try:
                        time.sleep(2)
                        next_button = wait.until(EC.element_to_be_clickable((By.XPATH, next_btn_xpath)))
                        next_button.click()
                        time.sleep(3)
                    except:
                        print("❌ No more 'Next' button or failed twice. Moving to next URL.")
                        break

            except Exception as e:
                print(f"❌ Error while scraping: {e}")
                break

    print(f"\n✅ Finished scraping {chapter_count} chapters from {url}. Data saved to '{output_file}'.")

# Done
driver.quit()
print("\n✅ All URLs processed!")
