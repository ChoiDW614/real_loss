# -*- coding: utf-8-sig -*-

import csv
import requests
from bs4 import BeautifulSoup

url = "https://finance.naver.com/sise/sise_market_sum.nhn?sosok=0&page="


filename = "C:/Users/djw04/Desktop/PythonQuant/PER_ROA.csv"
f = open(filename, "w", encoding="utf-8-sig", newline="")

writer = csv.writer(f)

for page in range(1, 5):
    
    res = requests.get(url + str(page))
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    
    # table 상단 넣어주기
    if(page==1):
        titles = soup.find("table", attrs={"class":"type_2"}).find("thead").find("tr")
        titles = titles.find_all("th")
        # print(titles)
        table_top = []
        for title in titles:
            table_top.append(title.string)
        #print(table_top)
        writer.writerow(table_top)

    data_rows = soup.find("table", attrs={"class":"type_2"}).find("tbody").find_all("tr")

    for row in data_rows:
        cols = row.find_all("td")
        if len(cols) <= 1:
            continue 

        data = [col.get_text().strip() for col in cols]
        writer.writerow(data)