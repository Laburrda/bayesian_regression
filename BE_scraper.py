from bs4 import BeautifulSoup
import json
from datetime import datetime
import undetected_chromedriver as uc
import requests
import time

class ScrapBrickEconomy:
    def __init__(self, tabs: list[str]) -> None:
        options = uc.ChromeOptions()
        options.headless = False
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-popup-blocking")

        self.driver = uc.Chrome(options=options)

        self.tabs: list[str] = tabs
        self.session = requests.Session()

        self.headers = {
            "User-Agent": self.driver.execute_script("return navigator.userAgent")
        }
        self.session.headers.update(self.headers)

    def get_soup(self, url: str):
        self.driver.get(url)
        time.sleep(10)

        cookies = {c["name"]: c["value"] for c in self.driver.get_cookies()}
        self.session.cookies.update(cookies)

        response = self.session.get(url, headers=self.headers)
        print(f"Status Code: {response.status_code}")

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        return soup

    def close(self):
        self.driver.quit()
    
    def modify_info(self, info_ls: list) -> dict:
        data = {}

        theme = info_ls[0].split('/')[-1]
        year = info_ls[1].split(' ')[-1]
        availability = info_ls[3].split(' ')[-1]

        data['theme'] = theme
        data['year'] = int(year)
        data['availability'] = availability

        elements = info_ls[2]
        if 'Pieces' in elements and 'Minifigs' in elements:
            temp_ls = elements.split(' ')

            pieces = temp_ls[3]
            pieces = pieces.replace(',', '')
            data['Pieces'] = int(pieces)

            minifigs = temp_ls[5]
            data['Minifigs'] = minifigs

            return data
        
        if 'Pieces' in elements:
            pieces = elements.split(' ')[-1]
            pieces = pieces.replace(',', '')
            data['Pieces'] = int(pieces)

            return data
        
        if 'Minifigs' in elements:
            minifigs = elements.split(' ')[-1]
            data['Minifigs'] = int(minifigs)

            return data

    def modify_prices(self, prices_ls) -> None:
        
        def _clear_value(string: str) -> float:
            raw_price = string.split(' ')[1]
            price = raw_price[1:]

            if price[0] == '$':
                price = raw_price[2:]
            
            if ',' in price:
                price = price.replace(',', '')

            price = float(price)

            return price

        data = {}
        if prices_ls[0][:3] == 'pro':
            data['Retail'] = 'Promotional'

            value = _clear_value(prices_ls[1])
            data['Value'] = value

            return data
        
        elif prices_ls[1][:1] == 'N':
            retail = _clear_value(prices_ls[0])
            data['Retail'] = retail

            data['Value'] = 'Not yet released'

            return data

        elif prices_ls[0] == '':
            data['Retail'] = 'Promotional or Unknown'

            value = _clear_value(prices_ls[1])
            data['Value'] = value

            return data

        else:
            retail = _clear_value(prices_ls[0])
            data['Retail'] = retail

            if prices_ls[1][:1] == 'A':
                data['Value'] = prices_ls[1]
            else:
                value = _clear_value(prices_ls[1])
                data['Value'] = value

            return data    
    
    def scrap_lego_set(self, lego_set):
        data = {}

        if not lego_set:
            return data
        
        def scrap_left_table(lego_set) -> str | list:
            left_table = lego_set.find('td', class_ = 'ctlsets-left')
            index = left_table.find('a', href = True)

            raw_info_ls = left_table.find_all('div', class_ = 'mb-2')
            in_info_ls = [info.text for info in raw_info_ls]
            info_ls = self.modify_info(in_info_ls)

            raw_stores = left_table.find_all('span', title=True)
            stores = [store.text for store in raw_stores]

            return index.text, info_ls, stores
        
        def scrap_right_table(lego_set):
            right_table = lego_set.find('td', class_ = 'ctlsets-right text-right')
            raw_info_ls = right_table.find_all('div')
            in_info_ls = [info.text for info in raw_info_ls[1:3]]
            info_ls =  self.modify_prices(in_info_ls)

            return info_ls

        index, data_ls, stores = scrap_left_table(lego_set)
        prices = scrap_right_table(lego_set)

        data['set_info'] = data_ls
        data['stores'] = stores
        data['prices'] = prices
        
        return index, data
    
    def get_main_table(self, soup: BeautifulSoup) -> list:
        data = []

        if not soup:
            return data

        main_table = soup.find('table', class_ = 'table table-hover ctlsets-table')

        sets = main_table.find_all('tr')

        for number, lego_set in enumerate(sets):
            if number % 2 == 0:
                index, set_data = self.scrap_lego_set(lego_set)
                temp_data = {index: set_data}
                data.append(temp_data)

        return data
    
    def scrape(self) -> list:
        data = []
        start_time = datetime.now()

        for tab in self.tabs:
            url = 'https://www.brickeconomy.com/sets/theme/star-wars/subtheme/'
            url = url + tab
            
            soup = self.get_soup(url)
            if not soup:
                print(f'Failed to scrape subtheme {tab} page.')
                continue
            
            set_info = self.get_main_table(soup)
            print(f'Successfully scraped subtheme {tab}. The data added to the output list.')

            data.extend(set_info)
        
        end_time = datetime.now()
        execition_time = end_time - start_time
        print(f'The total execution time was: {execition_time}')

        return data
    
    def write_json(self, data: dict) -> None:
        version = 'final'
        file_name = f'scraped_brickeconomy_{version}.json'

        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)

tabs = ['helmet-collection']

tab_ls = ['4-plus', 'ahsoka', 'andor', 'battlefront', 'book-parts', 'boost', 'buildable-figures',
          'comiccon', 'diorama-collection', 'employee-gift', 'episode-i', 'episode-ii', 'episode-iii',
          'episode-iv', 'episode-v', 'episode-vi', 'exclusive-minifigs', 'galaxys-edge', 'helmet-collection',
          'jedi-fallen-order', 'legends', 'master-builder-series', 'mechs', 'microfighters', 'miscellaneous',
          'original-content', 'planet-set', 'promotional', 'rebels', 'rebuild-the-galaxy', 'resistance', 
          'rogue-one', 'seasonal', 'skeleton-crew', 'solo', 'starship-collection', 'technic', 'the-bad-batch',
          'the-book-of-boba-fett', 'the-clone-wars', 'the-force-awakens', 'the-last-jedi', 'the-mandalorian',
          'the-old-republic', 'the-rise-of-skywalker', 'ultimate-collector-series', 'value-packs', 'young-jedi-adventures']

def main():
    obj = ScrapBrickEconomy(tab_ls)
    
    try:
        data = obj.scrape()
        obj.write_json(data)        
    finally:
        obj.close()

if __name__ == '__main__':
    main()

# The total execution time was: 0:08:41.812519























