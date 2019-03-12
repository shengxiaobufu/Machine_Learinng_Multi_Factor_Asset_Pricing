# 因子部分的数据预处理，综合基本面/宏观/行业三类因子，
# TODO 并滞后一期与y对应
# 对所有因子中包含的股票取股票代码的并集，去掉没有所有因子值的股票并输出

from config import config
import pandas as pd
import numpy as np
import os
from common_func import add_month

factor_pt = config.get('PATH', 'factors_pt')
indus_pt = config.get('PATH', 'data_pt') + '/indus/indus.csv'
macro_pt = config.get('PATH', 'data_pt') + '/macro/macro_all.csv'
deled_pt = config.get('PATH', 'data_pt') + '/factors_deled'
data_pt = config.get('PATH', 'data_pt')

rte_monthly = pd.read_csv(config.get('PATH', 'data_pt') + '/y.csv', index_col=0, parse_dates=True)
indus_monthly = pd.read_csv(indus_pt, index_col=0, parse_dates=True)
rte_monthly.columns = [int(i) for i in rte_monthly.columns]
indus_monthly.columns = [int(i) for i in indus_monthly.columns]
rte_monthly = rte_monthly.stack()
indus_monthly = indus_monthly.stack()

data_all = indus_monthly

factors_file = os.listdir(factor_pt)

for f in factors_file:
    if f.split('.')[1] != 'csv':
        continue
    print(f)
    factor = pd.read_csv(factor_pt + '/' + f, index_col=0, parse_dates=True)
    factor['code'] = factor['code'].astype(int).values
    factor = factor.reset_index().set_index(['year_month', 'code'])
    data_all = pd.concat([data_all, factor], axis=1).dropna(how='any')

macro_all = pd.read_csv(macro_pt, index_col=0, parse_dates=True)
assert len(macro_all) == 168
code_all = list(set(data_all.reset_index()['level_1'].values))
macro_with_code = None
for c in code_all:
    cs = [c] * 168
    macro_c = np.concatenate([np.array(cs).reshape(-1, 1), macro_all.values], axis=1)
    if macro_with_code is None:
        macro_with_code = macro_c
    else:
        macro_with_code = np.concatenate([macro_with_code, macro_c], axis=0)
macro_with_code = pd.DataFrame(macro_with_code, index=list(macro_all.index) * len(code_all),
                               columns=['code'] + ['macro-'+i for i in list(macro_all.columns)])
macro_with_code['code'] = macro_with_code['code'].astype(int)
macro_with_code = macro_with_code.reset_index().set_index(['index', 'code'])
print(macro_with_code.shape)

data_all = pd.concat([data_all, macro_with_code], axis=1)
data_all.columns = ['indus'] + list(data_all.columns)[1:]

# TODO 在这里把X滞后一期，然后与y对应concat
data_all = data_all.reset_index().set_index('level_0')
new_index = add_month(data_all.index)
data_all.index = new_index
data_all = data_all.reset_index().set_index(['index', 'level_1'])

data_all = pd.concat([rte_monthly, data_all], axis=1)
data_all = data_all[~np.isnan(data_all['indus'])]
data_all.columns = ['rte'] + list(data_all.columns)[1:]
data_all = data_all[~np.isnan(data_all['rte'])]
print(data_all.shape)
data_all.to_csv(data_pt + '/' + 'data_all.csv')



