from bs4 import BeautifulSoup
from requests_html import HTMLSession
import json
from functools import wraps
from datetime import datetime
import os

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

def calc_time(func: object) -> None:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()

        result = func(*args, **kwargs)

        end_time = datetime.now()
        execution_time = end_time - start_time
        print(f'The execution time is - {execution_time}.', flush=True)

        return result
    return wrapper

class ScrapSet:

    def __init__(self, 
                 page_url: str, 
                 headers: dict) -> None:
        self.page_url: str = page_url
        self.headers: dict = headers
        self.status_code: int = None
        self.session: HTMLSession = HTMLSession()
    
    def get_soup(self) -> BeautifulSoup | None:
        pageToScrape = self.session.get(self.page_url, headers=self.headers)

        if pageToScrape.status_code != 200:
            return None

        pageToScrape.html.render()
        soup = BeautifulSoup(pageToScrape.html.html, 'html.parser')

        return soup

    def scrap_logic(self,
                    soup: BeautifulSoup,
                    name: str, 
                    identifier: str, 
                    attr_type: str = "class") -> list:
        requested_ls = soup.find_all(name, attrs= {attr_type: identifier})

        out_ls = [element.text.replace(',', '') if ',' in element.text else element.text for element in requested_ls]

        return out_ls

    def scrap_basic_info(self,
                         soup: BeautifulSoup) -> dict:
        data = {}

        if not soup:
            return data
        
        out_ls = self.scrap_logic(soup, 'a', 'links')
        
        data['Year'] = int(out_ls[0])
        data['Parts'] = int(out_ls[1][:-6])

        for info in out_ls:
            if 'Minifigures' in info:
                data['Minifigures'] = int(info[:-12])
            elif 'Set' in info:
                data['In_sets'] = int(info[:-4])
    
        return data
    
    def scrap_weight(self,
                     soup: BeautifulSoup) -> dict:
        data = {}

        if not soup:
            return data
        
        out_ls = self.scrap_logic(soup, 'span', 'item-weight-info', attr_type = 'id')
        
        data['Weight'] = float(out_ls[0][:-1])

        return data
    
    def scrap_tables(self, 
                     soup: BeautifulSoup) -> dict:
        data = {}

        if not soup:
            return data
        
        tables = soup.find_all('table', class_='pcipgSummaryTable')

        for number, table in enumerate(tables):
            if number in (0, 2):
                try:
                    table_rows = table.find_all('tr')

                    tab = {}
                    for row in table_rows:
                        cols = row.find_all('td')
                        key = cols[0].text.strip()
                        value = cols[1].text.strip()

                        if ',' in value:
                            value = value.replace(',', '')

                        if 'PLN' in value:
                            value = float(value[4:])
                        else:
                            value = int(value)

                        tab[key] = value

                    key_name = "Past_table" if number == 0 else "Current_table"
                    data[key_name] = tab
                except:
                    continue

        return data

    @calc_time
    def return_dict(self) -> dict:
        soup = self.get_soup()

        data = {}
        if not soup:
            return data

        data.update(self.scrap_basic_info(soup))
        data.update(self.scrap_weight(soup))
        data.update(self.scrap_tables(soup))

        self.session.close()

        return data

class ScrapBricklink:

    def __init__(self, headers) -> None:
        self.headers: dict = headers
        self.total_pages: int = 20
        self.page_number: int = 1
        self.json_version: int | str = 'final'
        self.index_ls: list = []
        self.scraped_data: list = []
        self.failed_scrape: list = []
        self.session: HTMLSession = HTMLSession()


    def get_soup(self, url) -> BeautifulSoup | None:
        pageToScrape = self.session.get(url, headers=self.headers)

        if pageToScrape.status_code != 200:
            return None

        pageToScrape.html.render()
        soup = BeautifulSoup(pageToScrape.html.html, 'html.parser')

        return soup
    
    def get_index_list(self):

        for page_number in range(self.page_number, self.total_pages + 1):
            url = f'https://www.bricklink.com/catalogList.asp?pg={page_number}&catString=65&sortBy=Y&sortAsc=A&catType=S&v=1'
            soup = self.get_soup(url)

            if not soup:
                continue

            container = soup.find('div', class_='container-body')

            if container:
                self.index_ls.extend([a.text for a in container.find_all('a', href=True) if '-' in a.text])
        
        self.session.close()

        return self.index_ls
    
    def scrape_set_by_index(self) -> None:
        set_index_ls = self.get_index_list()

        start_time = datetime.now()
        scrapped_sets = 0
        failed_scraper = 0

        for set_index in set_index_ls:
            page_url = rf'https://www.bricklink.com/v2/catalog/catalogitem.page?S={set_index}#T=P'

            print(f'Program started scrapping set: {set_index}.')
            lego_set = ScrapSet(page_url, self.headers)
            
            try:
                set_data = lego_set.return_dict()

                if set_data:
                    self.scraped_data.append({set_index: set_data})
                    print(f'{set_index} has been succesfully scrapped.', '\n')

                scrapped_sets += 1
                print(f'Total number of scrapped sets: {scrapped_sets}.', '\n')
            except:
                self.failed_scrape.append(set_index)
                failed_scraper += 1
                print(f'Scrapper failed - set {set_index} the total number of failed sets: {failed_scraper}.', '\n')

        end_time = datetime.now()
        execution_time = end_time - start_time
        print(f'Total working time is {execution_time}.')    
    
    def write_json(self) -> None:
        
        self.scrape_set_by_index()

        file_name = f'scraped_bricklink_{self.json_version}.json'

        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(self.scraped_data, file, indent=4)
    
def main():
    lego_set = ScrapBricklink(headers)
    lego_set.write_json()

if __name__ == '__main__':
    main()

# final run - total time 1:42:22.016845