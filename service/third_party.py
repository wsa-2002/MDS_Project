from datetime import datetime, timedelta
import requests

import pandas as pd


def get_time_format(time: datetime) -> tuple[str, str]:
    return (time - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%S"), (time + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%S")


def get_average_temp(time: datetime, auth_token: str):
    timeFrom, timeTo = get_time_format(time)
    res = requests.get(
        f'https://opendata.cwb.gov.tw/api/v1/rest/datastore/F-D0047-061?Authorization={auth_token}&locationName=%E5%A4%A7%E5%AE%89%E5%8D%80&elementName=AT&sort=time&timeFrom={timeFrom}&timeTo={timeTo}').json()
    extract_data = res['records']['locations'][0]['location'][0]['weatherElement'][0]['time']
    new_df = pd.DataFrame(columns=['time', 'temperature'])
    for data in extract_data:
        tmp = pd.DataFrame(
            {'time': [data['dataTime']], 'temperature': [data['elementValue'][0]['value']]}, columns=['time', 'temperature'])
        new_df = pd.concat([new_df, tmp])
    df = pd.read_csv('data/temperature_record.csv')
    df = pd.concat([df[~df.time.isin(new_df.time)], new_df])
    df.to_csv('data/temperature_record.csv', index=False)
    return sum([int(data['elementValue'][0]['value']) for data in extract_data]) / 2


get_average_temp(
    datetime(2022, 12, 22, 19), auth_token='CWB-B6EDFA17-CB02-42EE-9213-366B3D6CCD0F')
