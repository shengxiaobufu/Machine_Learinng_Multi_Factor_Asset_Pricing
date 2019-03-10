# 数据预处理：剔除次新股/财务数据日频化

from config import config
import pandas as pd
import os
from datetime import date, datetime
from db_related import insert_many
import argparse


sql_daily = "INSERT IGNORE INTO `market_rt_daily` (`date`, `code`, `open`, `high`, `low`, `close`, `volume`," \
            "`amount`, `cap_liq`, `cap_all`, `rt`) " \
                      "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
sql_weekly = ''     # TODO weekly sql
sql_monthly = "INSERT IGNORE INTO `market_rt_monthly` (`date`, `code`, " \
              "`open`, `close`, `volume`, `amount`, `cap_liq`, `cap_all`, `trd_day_nums`, `rt`) " \
              "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"


def add_year(dates):
    dates_ = []
    for d in dates:
        year = d.year + 1
        month = d.month
        if d.month == 2 and d.day == 29:
            day = 28
        else:
            day = d.day
        dates_.append(date(year, month, day))
    return dates_


def add_6_months(dates):
    dates_ = []
    for d in dates:
        day = d.day
        if d.month <= 6:
            month = d.month + 6
            year = d.year
        else:
            year = d.year + 1
            month = d.month - 6
        try:
            dates_.append(date(year, month, day))
        except ValueError:
            day -= 1
            try:
                dates_.append(date(year, month, day))
            except ValueError:
                day -= 1
                try:
                    dates_.append(date(year, month, day))
                except ValueError:
                    day -= 1
                    dates_.append(date(year, month, day))
    return dates_


def del_cixin(start_date, rt, freq='monthly'):
    """要求index是date格式而不是datestamp"""
    assert freq in ['daily', 'weekly', 'monthly'], 'Unknown freq.'
    stocks = list(set(rt['Stkcd']))
    # out_rt = pd.DataFrame()
    for stock in stocks:
        try:
            s = start_date[start_date['Stkcd'] == stock].index[0]
        except:
            s = date(2005, 1, 1)
        print('Start date of', stock, 'is', s)
        rti = rt[rt['Stkcd'] == stock]
        # print(np.sum(np.isnan(rti)))
        # print([len(i) for i in rti if len(i) != 10])
        # rti = [list([str(j) for j in i]) for i in rti]
        if freq == 'daily':
            rti = list(rti[rti.index >= s].reset_index().values)
            rti = [list(i) for i in rti]
            insert_many(sql_daily, rti)
        elif freq == 'weekly':
            insert_many(sql_weekly, rti)
        elif freq == 'monthly':
            rti = list(rti[rti.index >= s].reset_index().set_index('Clsdt').values)
            rti = [list(i) for i in rti]
            insert_many(sql_monthly, rti)
    return


if __name__ == '__main__':
    data_pt = config.get('PATH', 'data_pt')
    market_data_pt = config.get('PATH', 'market_data_pt')
    ipo_date = pd.read_csv(data_pt + '/Ipoday_04-18.csv', encoding='utf8',
                           index_col=1, parse_dates=True)
    ipo_date.index = add_6_months(ipo_date.index)       # 直接给上市日期加上六个月

    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', default='monthly')
    args = parser.parse_args()
    freq = vars(args)['freq']

    rt_files = os.listdir(market_data_pt + '/' + freq)
    # 去掉次新股
    for f in rt_files:
        if f[-4:] == '.csv':
            print(f)
            # 以周度的close date作为基准日期
            rt = pd.read_csv(market_data_pt + '/' + freq + '/' + f, encoding='gb2312', index_col=1,
                             parse_dates=True)
            rt = rt.fillna('NULL')
            # rt = rt.iloc[:1000, :]
            if freq == 'monthly':
                rt.index = [datetime.strptime(i, '%b-%y') for i in rt.index]
                rt.index = [date(rt.index[i].year, rt.index[i].month,
                                 int(rt.iloc[i, 2])) for i in range(len(rt))]
                print('Start Deal!')
                del_cixin(ipo_date, rt, freq=freq)
            elif freq == 'daily':
                rt.index = [i.date() for i in rt.index]
                del_cixin(ipo_date, rt, freq=freq)







