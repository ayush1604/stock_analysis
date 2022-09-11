from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
import csv
from datetime import datetime, timedelta
import pandas as pd
from sklearn.utils import indices_to_mask


def get_sectoral_index(index, launch_date, csvfile=None):
    if csvfile is None:
        csvfile = index + '.csv'

    urlified_index = ''.join('%20' if c == ' ' else c for c in index)
    year = timedelta(days=364)
    start = datetime.fromisoformat(launch_date).date()
    flag = False
    driver = webdriver.Chrome()
    rows = []

    while not flag:
        start_date, end_date = start.strftime("%d-%m-%Y"), (start + year).strftime("%d-%m-%Y")
        if (start + year) > datetime.now().date():
            end_date = datetime.now().date().strftime("%d-%m-%Y")
            flag = True
        request_url = ("https://www1.nseindia.com//products/dynaContent/"
                       "equities/indices/historicalindices.jsp?indexType={}"
                       "&fromDate={}&toDate={}".format(urlified_index, start_date, end_date))
        driver.get(request_url)
        try:
            csvcontent = driver.find_element_by_id("csvContentDiv").get_attribute('innerHTML')
            #         print(start_date, end_date)
            for i_line, line in enumerate(csvcontent.split(':')):
                if i_line == 0:
                    continue
                row = []
                for cell in line.split(","):
                    row.append(cell.replace('"', '').strip())
                rows.append(row)
        except NoSuchElementException:
            print("Found no records for the period {} - {}".format(start_date, end_date))

        start += year + timedelta(days=1)

    df = pd.DataFrame(data=rows, columns=[
        "Date", "Open", "High", "Low", "Close", "SharesTraded", "Turnover"])
    df.to_csv(csvfile)
    print("{} saved.".format(csvfile))
    return df


if __name__ == "__main__":
    # indices_launch_date = {
    #     "NIFTY AUTO" : "2004-01-01",
    #     "NIFTY BANK" : "2000-01-01",
    #     "NIFTY CONSR DURBL" : "2005-04-01",
    #     "NIFTY FIN SERVICE" : "2004-01-01",
    #     "NIFTY FINSRV25 50" : "2004-01-01",
    #     "NIFTY FINANCIAL SERVICES EX-BANK": "2005-04-01",
    #     "NIFTY FMCG" : "1996-01-01",
    #     "NIFTY HEALTHCARE" : "2004-04-01",
    #     "NIFTY IT" : "1996-01-01",
    #     "NIFTY MEDIA" : "2005-12-30",
    #     "NIFTY METAL" : "2004-01-01",
    #     "NIFTY OIL AND GAS" : "2005-04-01",
    #     "NIFTY PHARMA" : "2001-01-01",
    #     "NIFTY PVT BANK" : "2004-05-01",
    #     "NIFTY PSU BANK" : "2004-01-01",
    #     "NIFTY REALTY" : "2006-12-29",
    # }

    indices_launch_date = {
        "NIFTY 50" : "1997-04-21"
    }

    for index, launch_date in indices_launch_date.items():
        get_sectoral_index(index, launch_date)