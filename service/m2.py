import util.Regression as Regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# 設定圖形大小; DPI越大圖越大
plt.rcParams["figure.dpi"] = 100

# myfont = FontProperties(fname=r'./NotoSansCJK-Light.ttc')
plt.rcParams['font.sans-serif'] = ['SimHei']


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def Emphasize(string, emphasis=''):
    if (emphasis == ''):
        return color.BOLD + string + color.END
    else:
        string1, string2 = string.split(emphasis)
        return color.BOLD + string1 + color.RED + emphasis + color.END + color.BOLD + string2 + color.END


class M2Model:
    def __init__(self):
        weekend = pd.date_range(start='2018-09-01', end='2022-08-31', freq='7D').append(
            (pd.date_range(start='2018-09-02', end='2022-08-31', freq='7D')))
        vacation = weekend.append(pd.date_range(start='2018-08-31', end='2018-09-10').append(pd.date_range(start='2019-01-13', end='2019-02-18')).append(pd.date_range(start='2019-06-24', end='2019-09-09')).append(pd.date_range(start='2020-01-13', end='2020-02-17')).append(pd.date_range(
            start='2020-06-22', end='2020-09-14')).append(pd.date_range(start='2021-01-18', end='2021-02-22')).append(pd.date_range(start='2021-06-28', end='2021-09-22')).append(pd.date_range(start='2022-01-26', end='2022-02-14')).append(pd.date_range(start='2022-06-20', end='2022-08-31')))
        vacation = vacation.append(pd.date_range(start='2018-08-31', end='2018-09-10').append(pd.date_range(start='2018-09-24', end='2018-09-24')).append(pd.date_range(start='2018-10-10', end='2018-10-10')).append(pd.date_range(start='2018-12-31', end='2019-01-01')).append(pd.date_range(start='2019-02-28', end='2019-03-1')).append(pd.date_range(start='2019-04-02', end='2019-04-05')).append(pd.date_range(start='2019-06-7', end='2019-06-7')).append(pd.date_range(start='2019-09-13', end='2019-09-13')).append(pd.date_range(start='2019-10-10', end='2019-10-11')).append(pd.date_range(start='2020-01-1', end='2020-01-1')).append(pd.date_range(start='2020-02-28', end='2020-02-28')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ).append(pd.date_range(start='2020-04-1', end='2020-04-6')).append(pd.date_range(start='2020-10-1', end='2020-10-2')).append(pd.date_range(start='2020-10-9', end='2020-10-9')).append(pd.date_range(start='2021-01-1', end='2021-01-1')).append(pd.date_range(start='2021-03-1', end='2021-03-1')).append(pd.date_range(start='2021-04-1', end='2021-04-6')).append(pd.date_range(start='2021-10-11', end='2021-10-11')).append(pd.date_range(start='2021-12-31', end='2021-12-31')).append(pd.date_range(start='2022-02-28', end='2022-02-28')).append(pd.date_range(start='2022-04-4', end='2022-04-5')).append(pd.date_range(start='2022-06-3', end='2022-06-3')))
        covid19_all = pd.date_range(start='2021-05-17', end='2021-10-12').append(
            (pd.date_range(start='2022-05-14', end='2022-08-15')))

        df = pd.read_excel('data/m2_data_no_outlier.xlsx')
        df = df.drop(columns='Unnamed: 0')
        df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d%H')
        df['HOUR'] = df['DATE'].dt.hour
        df['MONTH'] = df['DATE'].dt.month
        df['day'] = df['DATE'].dt.floor('D')

        df['WORK'] = 1
        df['WORK'] = df['day'].apply(lambda x: 0 if str(x) in vacation else 1)
        df['Covid_All'] = 1
        df['Covid_All'] = df['day'].apply(
            lambda x: 1 if str(x) in covid19_all else 0)

        temp = pd.read_excel(
            'data/temperature_imputation (knn).xlsx').drop(columns={'Unnamed: 0'})

        df_new = df.merge(temp, on='DATE', how='left').reset_index(drop=True)

        df_new['時段'] = df_new['HOUR'] // 3 + 1
        df_linear = df_new.groupby(['day', '時段'], as_index=False).agg(
            {'kWh': 'sum', '平均測站氣溫': 'mean', 'WORK': 'mean', 'Covid_All': 'mean', 'HOUR': 'count'})
        df_linear = df_linear[df_linear['HOUR'] == 3].drop(
            columns='HOUR').reset_index(drop=True)
        df_linear['index'] = df_linear.index

        df_train = df_linear[df_linear['day'] < '2021-09-01']
        df_test = df_linear[df_linear['day'] >=
                            '2021-09-01'].reset_index(drop=True)

        time_interval = list()
        for i in range(1, 9):
            time_interval.append('time_'+str(i))

        This = Regression.Regression_G(df_train, target_title={'y': 'kWh', 'x1': 'index', 'x2': '平均測站氣溫'},
                                       indicator={'qualitative': ['時段', 'WORK', 'Covid_All'], 'unit': [
                                           'time', 'work', 'covid_All'], 'name': [time_interval, ['非工作日', '工作日'], ['實體時期', '遠距時期']]},
                                       interaction=[['Covid_All', '平均測站氣溫'], ['time_3', '平均測站氣溫'], ['time_7', '平均測站氣溫'], ['time_2', '平均測站氣溫'], ['time_8', '平均測站氣溫'], ['time_1', '平均測站氣溫'], ['time_6', '平均測站氣溫'], ['time_5', '平均測站氣溫'], ['time_4', '平均測站氣溫'], ['WORK', '平均測站氣溫']])

        interaction = [['Covid_All', '平均測站氣溫'], ['time_3', '平均測站氣溫'], ['time_7', '平均測站氣溫'], ['time_2', '平均測站氣溫'], ['time_8', '平均測站氣溫'], [
            'time_1', '平均測站氣溫'], ['time_6', '平均測站氣溫'], ['time_5', '平均測站氣溫'], ['time_4', '平均測站氣溫'], ['WORK', '平均測站氣溫']]
        tmp = pd.get_dummies(df_test['時段'], prefix='time')
        df_test = pd.concat([df_test, tmp], axis=1)
        df_test[''] = 1
        for j in interaction:
            df_test[j[0] + '_' + j[1]] = df_test[j[0]] * df_test[j[1]]
        df_forcast = pd.DataFrame({'day': df_test['day'], 'true': df_test['kWh'],
                                   'predict': This.result.predict(df_test[This.x_name].values)})

        df_forecast = pd.DataFrame()
        df_forecast['DATE'] = pd.date_range(
            start='2022-09-01', end='2023-08-31')
        df_forecast = df_forecast.loc[df_forecast.index.repeat(
            8)].reset_index(drop=True)
        df_forecast['時段'] = df_forecast.index % 8+1
        weekend2 = pd.date_range(start='2022-09-03', end='2023-08-31', freq='7D').append(
            (pd.date_range(start='2022-09-04', end='2023-08-31', freq='7D')))
        vacation2 = weekend2.append(pd.date_range(start='2022-09-9', end='2022-09-9').append(pd.date_range(start='2022-10-10', end='2022-10-10')).append(pd.date_range(start='2022-12-26', end='2023-2-17')
                                                                                                                                                         ).append(pd.date_range(start='2023-2-27', end='2023-2-28')).append(pd.date_range(start='2023-4-3', end='2023-4-5')).append(pd.date_range(start='2023-6-12', end='2023-8-31')))
        df_forecast['WORK'] = 1
        df_forecast['Covid_All'] = 0
        df_forecast['Covid_60'] = 0
        df_forecast['WORK'] = df_forecast['DATE'].apply(
            lambda x: 0 if str(x) in vacation2 else 1)
        tmp = pd.get_dummies(df_forecast['時段'], prefix='time')
        df_forecast = pd.concat([df_forecast, tmp], axis=1)
        df_forecast[''] = 1
        df_forecast['index'] = np.arange(
            len(df_linear), len(df_linear) + len(df_forecast))

        self.model = This
        self.df_forecast = df_forecast

    def predict(self, future_temp, dateTime):
        model = self.model
        df_forecast = self.df_forecast
        dateTime = pd.to_datetime(dateTime, format='%Y-%m-%d %H')
        time_hour = dateTime.hour
        interval = time_hour // 3 + 1
        time_date = dateTime.date()
        p = df_forecast[(df_forecast['DATE'] == str(time_date))
                        & (df_forecast['時段'] == interval)]
        p['平均測站氣溫'] = future_temp
        interaction = [['Covid_All', '平均測站氣溫'], ['time_3', '平均測站氣溫'], ['time_7', '平均測站氣溫'], ['time_2', '平均測站氣溫'], ['time_8', '平均測站氣溫'], [
            'time_1', '平均測站氣溫'], ['time_6', '平均測站氣溫'], ['time_5', '平均測站氣溫'], ['time_4', '平均測站氣溫'], ['WORK', '平均測站氣溫']]
        for j in interaction:
            p[j[0] + '_' + j[1]] = p[j[0]] * p[j[1]]
        value_forecast = pd.DataFrame(
            {'day': p['DATE'], 'predict': model.result.predict(p[model.x_name].values)})
        return value_forecast


m2_model = M2Model()
