import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from service.crawler import get_elec_of_day
from service.m1 import m1_model
from service.m2 import m2_model
from service.third_party import get_average_temp
from config import auth_token

import warnings
warnings.filterwarnings('ignore')


def get_past_day_elec_usage(date: datetime.datetime, days=7, ctg='04C_P1_01'):
    df = pd.DataFrame()
    usages = []
    for i in range(days, -1, - 1):
        day = date - datetime.timedelta(days=i)
        usage = get_elec_of_day(day, ctg=ctg)
        usages.extend(usage[1:])
    df['usage'] = usages
    df['date'] = [datetime.datetime(date.year, date.month, date.day - day, hour)
                  for day in range(days, -1, -1) for hour in range(0, 24)]
    return df


print("app reload...")
# 設定網頁標題
st.title('台大電量即時觀測系統')
m1, m2 = st.tabs(["管一", "管二"])


with m1:
    st.header(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    col1, col2 = st.columns(2)
    now = datetime.datetime.now()

    with col1:
        day_elec = get_elec_of_day(now.date())
        col1.metric("current usage", value=sum(
            day_elec[now.hour-2:now.hour+1]))

    with col2:
        future_temp = get_average_temp(now, auth_token=auth_token)
        prediction = m1_model.predict(future_temp=future_temp, dateTime=now)
        col2.metric('predicted usage', prediction['predict'].values[0])

    st.header('previous electricity usage')

    col1, col2 = st.columns(2)
    with col1:
        start_time = st.date_input(
            "start_time", (datetime.date(2022, 8, 31) - datetime.timedelta(days=7)))
    with col2:
        end_time = st.date_input("end_time", datetime.date(2022, 8, 31))
    previous_usage = get_past_day_elec_usage(
        now.date(), days=(end_time-start_time).days)
    st.line_chart(previous_usage, x='date', y='usage')
    st.button('refresh', key='m1')

with m2:
    st.header(datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))

    col1, col2 = st.columns(2)
    with col1:
        now = datetime.datetime.now()
        day_elec = get_elec_of_day(now.date(), "05A_P1_02")
        st.metric("current usage", value=sum(
            day_elec[now.hour-2:now.hour+1]))

    with col2:
        future_temp = get_average_temp(now, auth_token=auth_token)
        prediction = m2_model.predict(future_temp=future_temp, dateTime=now)
        st.metric('predicted usage', prediction['predict'].values[0])

    st.header('previous electricity usage')
    col1, col2 = st.columns(2)
    with col1:
        start_time = st.date_input(
            "start_time", (datetime.date(2022, 8, 31) - datetime.timedelta(days=7)), key='m2_start')
    with col2:
        end_time = st.date_input(
            "end_time", datetime.date(2022, 8, 31), key='m2_end')
    previous_usage = get_past_day_elec_usage(
        now.date(), ctg="05A_P1_02", days=(end_time-start_time).days)
    st.line_chart(previous_usage, x='date', y='usage')
    st.button('refresh', key='m2')
