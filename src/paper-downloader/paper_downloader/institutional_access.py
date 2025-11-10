from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pickle, time

# --- configure browser ---
options = Options()
# options.add_argument("--headless=new")  # omit if you want to see the window
driver = webdriver.Chrome(options=options)

# --- navigate to ACS article ---
acs_url = "https://pubs.acs.org/doi/10.1021/acs.chemmater.5b02557"
driver.get(acs_url)

print("🔐 Log in with your Georgia Tech credentials if prompted...")
input("Press Enter once you see the full article page loaded...")

# --- save cookies after successful login ---
cookies = driver.get_cookies()
with open("acs_cookies.pkl", "wb") as f:
    pickle.dump(cookies, f)

print(f"Saved {len(cookies)} cookies for ACS session.")
driver.quit()