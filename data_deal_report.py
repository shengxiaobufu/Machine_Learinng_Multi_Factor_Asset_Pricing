# 数据预处理：剔除次新股/财务数据日频化

from config import config
import pandas as pd
import numpy as np
import os
from datetime import date
from db_related import insert_many, get_connect
import codecs
import warnings

warnings.filterwarnings('ignore')


def get_columns_name(path, start='A'):
    """读取字段说明txt，获取财报会计科目列名"""
    # path = r'C:\Users\bjzjl\Desktop\jianzhi\Machine_Learinng_Multi_Factor_Asset_Pricing\data\report\字段说明_equity_chg.txt'
    names = []
    with codecs.open(path, 'r', encoding='utf8') as f:
        line = f.readline()
        while line:
            name_temp = line.split(' ')[0]
            if name_temp[0] == start and name_temp != 'Accper':
                names.append(name_temp)
            line = f.readline()
    sql_str = '(`date`, `code`, `' + '`,`'.join(names) + '`) VALUES(%s, %s,' + ','.join(['%s' for i in names]) + ')'
    # create_str = ''
    # for n in names:
    #     create_str += n + ' FLOAT(50),'
    return sql_str


def match_report_date(report, dates):
    """匹配财务报表及其发布时间"""
    index = []
    for i in range(len(report)):
        acc_date = date(report.index[i].year, report.index[i].month, 1)
        true_date = dates.index[dates['Accper'] == acc_date]
        if len(true_date) != 1:
            if acc_date.month <= 10:
                true_date = date(acc_date.year, acc_date.month+2, 20)
            else:
                true_date = date(acc_date.year + 1, acc_date.month - 10, 20)
            # print(true_date)
        else:
            true_date = true_date[0].date()
        index.append(true_date)
    report.index = index
    return report


def fill_report_data(report, report_date, date_all, type):
    assert type in ['bs', 'cf_d', 'cf_ind', 'eq', 'pro'], 'Unknown report type.'
    stocks = list(set(date_all[:, 0]))
    # out_re = pd.DataFrame()
    for stock in stocks:
        print(stock)
        rt_date_ = [i.date() for i in date_all[:, 1][date_all[:, 0] == stock]]
        report_ = report[report['Stkcd'] == stock]
        anno_dates_ = report_date[report_date['Stkcd'] == stock]
        report_ = match_report_date(report_, anno_dates_)
        # 考虑到有年报和一季报一起发的情况，此时将直接删掉年报，保留一季报数据
        drop_dupli = report_.index[::-1].duplicated()[::-1]
        if np.sum(drop_dupli) > 0:
            print('Find duplicate in stock', stock, ', drop', np.sum(drop_dupli), 'rows.')
        report_ = report_.iloc[~drop_dupli, :]
        report_ = pd.concat([pd.DataFrame(rt_date_, index=rt_date_), report_], axis=1).iloc[:, 1:]
        report_ = report_.fillna(method='pad')
        # 把多余的日期删掉
        date_remain = [d in rt_date_ for d in report_.index]
        report_ = report_.iloc[date_remain, :].reset_index().set_index('Typrep').fillna('NULL')
        # report_ = report_.astype(object).where(pd.notnull(report_), None)
        report_ = list(report_.values)
        report_ = [list(i) for i in report_]
        if type_ == 'bs':
            insert_many(sql_bs, report_)
        elif type_ == 'cf_d':
            insert_many(sql_cf_d, report_)
        elif type_ == 'cf_ind':
            insert_many(sql_cf_ind, report_)
        elif type_ == 'eq':
            insert_many(sql_eq_cg, report_)
        elif type_ == 'pro':
            insert_many(sql_pro, report_)
    return


if __name__ == '__main__':
    data_pt = config.get('PATH', 'data_pt')
    market_data_pt = config.get('PATH', 'market_data_pt')
    report_data_pt = config.get('PATH', 'report_data_pt')

    sql_bs = "INSERT IGNORE INTO `report_bs_daily` " \
             + get_columns_name(report_data_pt + '/字段说明_bs.txt', start='A')
    sql_pro = "INSERT IGNORE INTO `report_profit_daily` " \
               + get_columns_name(report_data_pt + '/字段说明_profit.txt', start='B')
    sql_cf_d = "INSERT IGNORE INTO `report_cf_direct_daily` " \
               + get_columns_name(report_data_pt + '/字段说明_cf_direct.txt', start='C')
    sql_cf_ind = "INSERT IGNORE INTO `report_cf_indirect_daily` " \
                 + get_columns_name(report_data_pt + '/字段说明_cf_indirect.txt', start='D')
    sql_eq_cg = "INSERT IGNORE INTO `report_eq_change_daily` " \
                + get_columns_name(report_data_pt + '/字段说明_equity_chg.txt', start='F')

    report_files = os.listdir(report_data_pt)
    report_date = pd.read_csv(data_pt + '/report_date.csv', index_col=3, parse_dates=True,
                              encoding='gb2312')
    report_date['Accper'] = [date(int(i.split('/')[0]), int(i.split('/')[1]), 1) for
                             i in report_date['Accper']]

    conn = get_connect()
    with conn.cursor() as cursor:
        print('getting rt data...')
        cursor.execute('SELECT `date`, `code` FROM `market_rt_daily`')
        conn.commit()
        date_all = pd.DataFrame(cursor.fetchall()).values
    conn.close()

    # 财报数据日频化
    for f in report_files:
        if f[-4:] == '.csv':
            type_ = f.split('.')[0]
            if type_ in ['bs', 'cf_d', 'cf_ind', 'eq']:     # TODO eq表格式和其他表完全不一样，如无必要不处理
                continue        # TODO 其他三个要删掉
            print(type_)
            report = pd.read_csv(report_data_pt + '/' + f, encoding='gb2312', index_col=1,
                                 parse_dates=True)
            report = report[~np.isnan(report['Stkcd'])]
            # 这里应该用合并报表A，并去掉年报，保留季报数据
            report = report[(report['Typrep'] == 'A') * (report.index.month != 1)]
            report.index = [date(i.year, i.month, 1) for i in report.index]
            # 财报数据应以公布日期，而非会计期间为准，这里需要根据公布日期数据进行调整
            fill_report_data(report, report_date, date_all=date_all, type=type_)

