from datetime import datetime
import requests

from bs4 import BeautifulSoup
import numpy as np

NTU_elec_endpoint = 'https://epower.ga.ntu.edu.tw/fn4/report5.aspx?ctg=04C_P1_01&dt1=2022/12/22&ok=%BDT%A9w'


HOUR_TO_INDEX_MAP = {1: 15,
                     2: 16,
                     3: 17,
                     4: 18,
                     5: 19,
                     6: 20,
                     7: 21,
                     8: 22,
                     9: 23,
                     10: 24,
                     11: 25,
                     12: 26,
                     13: 41,
                     14: 42,
                     15: 43,
                     16: 44,
                     17: 45,
                     18: 46,
                     19: 47,
                     20: 48,
                     21: 49,
                     22: 50,
                     23: 51,
                     24: 52}


def get_elec_of_day(date: datetime, ctg: str = "04C_P1_01"):
    """
    ctg: 04C_P1_01 or 05A_P1_02
    """
    res = requests.post(
        f"https://epower.ga.ntu.edu.tw/fn4/report5.aspx?ctg={ctg}&dt1={date}&ok=%BDT%A9w")
    soup = BeautifulSoup(res.text, "html.parser")
    tds = soup.find("table", {"class": "style3"}).find_all("td")
    day_usage = np.zeros(25)
    for hour, index in HOUR_TO_INDEX_MAP.items():
        try:
            usage = float(tds[index].text)
        except:
            usage = None
        day_usage[hour] = usage
    return day_usage
