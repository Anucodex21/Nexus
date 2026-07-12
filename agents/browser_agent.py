from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

class BrowserAgent:
    """Agent that can browse and interact with web pages."""

    def __init__(self, headless=True):
        self.headless = headless
        self.driver = None
        self.history = []

    def _init_driver(self):
        """Initialize the browser driver."""
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(options=options)

    def navigate(self, url):
        """Navigate to a URL."""
        if not self.driver:
            self._init_driver()

        self.driver.get(url)
        self.history.append({"action": "navigate", "url": url})
        time.sleep(2)
        return self.get_page_content()

    def get_page_content(self):
        """Get the current page content."""
        if not self.driver:
            return None

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator='\n', strip=True)
        return text[:5000]  # Limit content length

    def click_element(self, selector, by=By.CSS_SELECTOR):
        """Click an element on the page."""
        if not self.driver:
            return False

        try:
            element = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((by, selector))
            )
            element.click()
            self.history.append({"action": "click", "selector": selector})
            return True
        except Exception as e:
            print(f"Click failed: {e}")
            return False

    def search(self, query, search_box_selector="input[type='search']"):
        """Search on the current page."""
        if not self.driver:
            return None

        try:
            search_box = self.driver.find_element(By.CSS_SELECTOR, search_box_selector)
            search_box.send_keys(query)
            search_box.submit()
            time.sleep(2)
            return self.get_page_content()
        except Exception as e:
            print(f"Search failed: {e}")
            return None

    def close(self):
        """Close the browser."""
        if self.driver:
            self.driver.quit()
            self.driver = None
