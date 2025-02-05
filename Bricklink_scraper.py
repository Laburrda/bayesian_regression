from bs4 import BeautifulSoup
from requests_html import HTMLSession
import json
from functools import wraps
from datetime import datetime
import os

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

#TODO: Create outer class to scrap all sets in a loop -> check urls
#TODO: Add response condition

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
        self.set_info: dict = {}
        self.set_index: str = ''
        self.status_code: int = None
        self.session: HTMLSession = HTMLSession()
    
    def get_soup(self) -> BeautifulSoup | None:
        pageToScrape = self.session.get(self.page_url, headers=self.headers)

        if pageToScrape.status_code != 200:
            self.status_code = None
        else:
            self.status_code = pageToScrape.status_code
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
        out_ls = self.scrap_logic(soup, 'a', 'links')
        
        self.set_info['Year'] = int(out_ls[0])
        self.set_info['Parts'] = int(out_ls[1][:-6])

        for info in out_ls:
            if 'Minifigures' in info:
                self.set_info['Minifigures'] = int(info[:-12])
            elif 'Set' in info:
                self.set_info['In_sets'] = int(info[:-4])
            else:
                pass
    
        return self.set_info
    
    def scrap_tables(self, 
                     soup: BeautifulSoup) -> dict:
        tables = soup.find_all('table', class_='pcipgSummaryTable')

        for number, table in enumerate(tables):
            if number in (0, 2):
                try:
                    table_rows = table.find_all('tr')

                    data = {}
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

                        data[key] = value
                except:
                    data = {}

            match number:
                case 0:
                    self.set_info['Past_table'] = data
                case 2:
                    self.set_info['Current_table'] = data

        return self.set_info
    
    def scrap_weight(self,
                     soup: BeautifulSoup) -> dict:
        out_ls = self.scrap_logic(soup, 'span', 'item-weight-info', attr_type = 'id')
        
        self.set_info['Weight'] = float(out_ls[0][:-1])

        return self.set_info

    def check_status_code(self) -> None:
        soup = self.get_soup()

        return soup
    
    @calc_time
    def return_dict(self) -> dict:
        soup = self.check_status_code()

        set_dict = self.scrap_basic_info(soup)
        set_dict = self.scrap_weight(soup)
        set_dict = self.scrap_tables(soup)
        self.session.close()

        return set_dict

class ScrapBricklink:

    def __init__(self, headers) -> None:
        self.headers: dict = headers
        self.total_pages: int = 20
        self.page_number: int = 1
        self.starting_page: str = f'https://www.bricklink.com/catalogList.asp?pg={self.page_number}&catString=65&sortBy=Y&sortAsc=A&catType=S&v=1'
        self.current_page: str = self.starting_page
        self.index_ls: list = []
        self.scraped_data: list = []
        self.failed_scrape: list = []
        self.session: HTMLSession = HTMLSession()


    def get_soup(self) -> BeautifulSoup:
        pageToScrape = self.session.get(self.current_page, headers=self.headers)

        if pageToScrape.status_code != 200:
            self.soup = None
        else:
            pageToScrape.html.render()
            self.soup = BeautifulSoup(pageToScrape.html.html, 'html.parser')

        return self.soup
    
    def scrape_page(self) -> None:
        soup = self.get_soup()

        page_containter = soup.find('div', class_ = 'container-xl container-body l-pad-y l-margin-bottom catalog-list__body')
        indexes = page_containter.find_all('a', href=True)
        indexes_ls = [index.text for index in indexes if '-' in index.text]
        
        self.index_ls.extend(indexes_ls)
    
    def scrap_all_pages(self) -> None:

        while self.page_number <= self.total_pages:
            self.scrape_page()
            self.page_number += 1
            self.current_page = f'https://www.bricklink.com/catalogList.asp?pg={self.page_number}&catString=65&sortBy=Y&sortAsc=A&catType=S&v=1'
        
        self.session.close()

    def get_index_list(self) -> list:
        self.scrap_all_pages()
        index_ls = self.index_ls

        return index_ls 
    
    def scrape_set_by_index(self) -> None:
        set_index_ls = self.get_index_list()

        start_time = datetime.now()
        scrapped_sets = 0
        failed_scraper = 0

        for set_index in set_index_ls:
            temp_dict = {}
            page_url = rf'https://www.bricklink.com/v2/catalog/catalogitem.page?S={set_index}#T=P'

            print(f'Program started scrapping set: {set_index}.')
            lego_set = ScrapSet(page_url, self.headers)
            lego_set.check_status_code()

            if lego_set.status_code != 200:
                print(f'Scrapper failed - set: {set_index} status code: {lego_set.status_code}.')
                continue
            else:
                try:
                    set_dict = lego_set.return_dict()
                    temp_dict[set_index] = set_dict
                    self.scraped_data.append(temp_dict)
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

        json_file = self.scraped_data
        file_name = 'scraped_bricklink.json'

        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(json_file, file, indent=4)
    
def main():
    bricklink = ScrapBricklink(headers)
    bricklink.write_json()

if __name__ == '__main__':
    main()

# final run - total time 2:09:43.548195