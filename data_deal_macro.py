# 宏观因子数据处理

from config import config
import pandas as pd
import numpy as np
from datetime import datetime, date
from data_deal_func import fullfill_monthly, daily2monthly
from db_related import select_data


macro_pt = config.get('PATH', 'data_pt') + '/macro'

# CPI
cpi = pd.read_csv(macro_pt + '/CPI.csv', index_col=0, parse_dates=True)
cpi = cpi.iloc[:, 1][cpi.iloc[:, 0] == 'A']
cpi.index = [datetime.strptime(i, '%b-%y').date() for i in cpi.index]
cpi = fullfill_monthly(2005, 2018, cpi)

# 1y,10y国债利率，期限利差
gov_rt = pd.read_csv(macro_pt + '/gov_bond_rt.csv', index_col=0, parse_dates=True)
gov_rt_1y = gov_rt['Yield'][gov_rt['Yeartomatu'] == 1]
gov_rt_10y = gov_rt['Yield'][gov_rt['Yeartomatu'] == 10]
gov_rt_ls = pd.concat([gov_rt_1y.rename('rt_1y'), gov_rt_10y.rename('rt_10y')], axis=1).fillna(method='pad')
gov_rt_ls['term_spread'] = gov_rt_ls['rt_10y'].values - gov_rt_ls['rt_1y'].values
gov_rt_ls = gov_rt_ls/100
mean_rt = daily2monthly(gov_rt_ls, group_c=['year_month'])
mean_rt.index = [datetime.strptime(i, '%Y-%m').date() for i in mean_rt.index]
print(mean_rt.shape)
mean_rt = fullfill_monthly(2005, 2018, mean_rt)
print(mean_rt.shape)
del gov_rt_ls
del gov_rt
del gov_rt_1y
del gov_rt_10y


# 市场EP/BM, 信用利差，NTIS
security_related = pd.read_excel(macro_pt + '/macro_pe_pb_crespr_ntis.xls', encoding='utf8',
                                 index_col=0, parse_dates=True).iloc[1:-2, :]
security_related['筹资金额:A股增发:当月值'] = security_related['筹资金额:A股增发:当月值']\
                                            .fillna(method='bfill')
security_related = security_related.fillna(method='pad')
security_related = security_related.iloc[:, [0, 1, 5, 6, 7]]
security_related.index = [datetime.strptime(i.split(' ')[0], '%Y-%m-%d')
                          for i in security_related.index]
mean_sec_rela = daily2monthly(security_related, group_c=['year_month'])
mean_sec_rela.index = [datetime.strptime(i, '%Y-%m').date() for i in mean_sec_rela.index]
print(mean_sec_rela.shape)
mean_sec_rela = fullfill_monthly(2005, 2018, mean_sec_rela)
print(mean_sec_rela.shape)
mean_sec_rela['ntis'] = mean_sec_rela.iloc[:, 1].values / mean_sec_rela.iloc[:, 0].values
mean_sec_rela = mean_sec_rela.iloc[:, 2:]
mean_sec_rela.iloc[:, :2] = 1 / mean_sec_rela.iloc[:, :2]
mean_sec_rela.columns = ['market_ep', 'market_bm', 'credit_spread', 'ntis']
del security_related

# 组合所有宏观因子
assert np.sum(~(cpi.index == mean_rt.index)) == 0
assert np.sum(~(mean_sec_rela.index == mean_rt.index)) == 0
macro_all = pd.DataFrame(np.concatenate([cpi.values, mean_rt.values, mean_sec_rela.values], axis=1),
                         index=cpi.index,
                         columns=list(cpi.columns) + list(mean_rt.columns)
                                 + list(mean_sec_rela.columns))
macro_all.to_csv(macro_pt + '/macro_all.csv')

# 用无风险利率处理月度收益率，获得超额收益
select_sql_monthly = 'SELECT `date`, `code`, `rt`, `trd_day_nums` FROM `market_rt_monthly`' \
                     ' WHERE `code` != 0'
monthly_rt = select_data(select_sql_monthly).reset_index().set_index('date').iloc[:, 1:]
monthly_rt = monthly_rt[monthly_rt['trd_day_nums'] >= 15]
monthly_rt = monthly_rt.iloc[:, :2]
# 在这里过滤交易天数，月交易日不足15天的都将被过滤掉

monthly_rt.index = [date(i.year, i.month, 1) for i in monthly_rt.index]
monthly_rt_pivoted = pd.pivot_table(monthly_rt, values='rt', index=monthly_rt.index, columns='code')
monthly_rt_pivoted_filled = fullfill_monthly(2005, 2018, monthly_rt_pivoted)
assert len(monthly_rt_pivoted_filled) == 168
rf = pd.read_csv(macro_pt + '/rf.csv', index_col=0, parse_dates=True)
rf.index = [i.date() for i in rf.index]
rf_monthly = daily2monthly(rf/100, group_c='year_month').iloc[:, -1]
assert len(rf_monthly) == 168
rte_monthly = monthly_rt_pivoted_filled - np.array([rf_monthly.values] * 3598).T
rte_monthly.to_csv(config.get('PATH', 'data_pt') + '/y.csv')


