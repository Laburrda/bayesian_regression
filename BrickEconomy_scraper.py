from bs4 import BeautifulSoup
from requests_html import HTMLSession
import json
import cloudscraper
from functools import wraps
from datetime import datetime
import os

#TODO: create BrickEconomy scraper
#TODO: add missing 51th set from episode-vi + second page at clone wars
#TODO: finish tab ls + obi wan kenobi cannot be loaded


tab_ls = ['4-plus', 'ahsoka', 'andor', 'battlefront', 'book-parts', 'boost', 'buildable-figures',
          'comiccon', 'diorama-collection', 'employee-gift', 'episode-i', 'episode-ii', 'episode-iii',
          'episode-iv', 'episode-v', 'episode-vi', 'exclusive-minifigs', 'galaxys-edge', 'helmet-collection',
          'jedi-fallen-order', 'legends', 'master-builder-series', 'mechs', 'microfighters', 'miscellaneous',
          'original-content', 'planet-set', 'promotional', 'rebels', 'rebuild-the-galaxy', 'resistance', 
          'rogue-one', 'seasonal', 'skeleton-crew', 'solo', 'starship-collection', 'technic', 'the-bad-batch',
          'the-book-of-boba-fett', 'the-clone-wars', 'the-force-awakens', 'the-last-jedi', 'the-mandalorian',
          'the-old-republic', 'the-rise-of-skywalker', 'ultimate-collector-series', 'value-packs', 'young-jedi-adventures']

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

class ScrapBrickEconomy:

    def __init__(self, headers: dict, tabs: list[str], url: str) -> None:
        self.headers: dict = headers
        self.tabs: list[str] = tabs
        self.url: str = url
        # self.session: HTMLSession = HTMLSession()

    def get_soup(self, url: str) -> BeautifulSoup | None:
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
 
        return soup
    
    def scrap_lego_set(self, lego_set):
        
        data = {}
        scrapped_ls = []

        if not lego_set:
            return data
        
        def scrap_left_table(lego_set):
            left_table = lego_set.find('td', class_ = 'ctlsets-left')
            index = left_table.find('a', href = True)

            raw_info_ls = left_table.find_all('div', class_ = 'mb-2')
            info_ls = [info.text for info in raw_info_ls]

            raw_stores = left_table.find_all('span', title=True)
            stores = [store.text for store in raw_stores]
            info_ls.extend(stores)

            return index.text, info_ls
        
        def scrap_right_table(lego_set):
            right_table = lego_set.find('td', class_ = 'ctlsets-right text-right')
            print(right_table)
            raw_info_ls = right_table.find_all('div', class_ = 'text-muted mr-5')
            info_ls = [info.text for info in raw_info_ls]
            print(info_ls)

            return info_ls

        index, data_ls = scrap_left_table(lego_set)
        scrapped_ls.extend(data_ls)

        data_ls = scrap_right_table(lego_set)
        scrapped_ls.extend(data_ls)

        data[index] = scrapped_ls
        
        return data
    
    def get_main_table(self, soup: BeautifulSoup) -> list:
        data = {}
        help_ls = []
        if not soup:
            return data

        main_table = soup.find('table', class_ = 'table table-hover ctlsets-table')
        sets = main_table.find_all('tr')

        for number, lego_set in enumerate(sets[:2]):
            if number % 2 == 0:
                index = self.scrap_lego_set(lego_set)
                help_ls.append(index)

        return help_ls
    
    def execute(self):
        soup = self.get_soup(self.url)

        table = self.get_main_table(soup)

        return table

# tbody -> tr[co dwa] -> td [1:2] -> td[1] , div[:5] | 
test_url = 'https://www.brickeconomy.com/sets/theme/star-wars/subtheme/the-force-awakens'

def main():
    obj = ScrapBrickEconomy(headers, tab_ls, test_url)
    table = obj.execute()

    print(len(table))
    for i in table:
        print(i)

if __name__ == '__main__':
    main()

























