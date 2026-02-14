# https://taiwanindex.com.tw/indexes/IR0174

import requests
from bs4 import BeautifulSoup
import re

# URL of the webpage you want to scrape
urls = {"bank": "https://www.moneydj.com/z/zh/zha/zh00.djhtm?a=C028100",
}

class IndustryStockList():
    def __init__(self):
        industry = input("Select a industry: [bank, .], default bank") or "bank"
        self.url = urls[industry]
        
    def check_format(self, s):
        pattern = r'^\d{4}\D+$'  # 4 digits followed by only letters
        return bool(re.match(pattern, s))

    def getList(self):        
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(self.url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            names = []
            for i in soup.find_all(id="oAddCheckbox"):
                if self.check_format(i.text):
                    names.append(i.text[:4] + ".TW")
            print(names)
        else:
            print(f"Failed to retrieve page, status code: {response.status_code}")

if __name__ == "__main__":
    scraper = IndustryStockList()
    scraper.getList()
