# 给出数据，方便统一修改所有模型使用的数据
# TODO 去掉17/18的数据再试试
import sys
sys.path.append("..")

from config import config
import pandas as pd
import numpy as np
import os
from test_frame import FactorTest
import warnings
from datetime import datetime
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')

#############################################################
n = 5
weight = 'ave'
#############################################################

factor_pt = config.get('PATH', 'factors_pt')
data_pt = config.get('PATH', 'data_pt')
out_pt = config.get('PATH', 'out_pt')

rte_monthly = pd.read_csv(config.get('PATH', 'data_pt') + '/market/monthly/rt_monthly_05-18.csv',
                          index_col=1, parse_dates=True)
rte_monthly.index = [datetime.strptime(i, '%b-%y') for i in rte_monthly.index]
rte_monthly = pd.pivot_table(rte_monthly, index=rte_monthly.index, columns='Stkcd', values='Mretwd')
rte_monthly = rte_monthly.iloc[rte_monthly.index.year > 2007, :]
# rte_monthly = rte_monthly.iloc[rte_monthly.index.year < 2017, :]        # TODO
cap = pd.read_csv(factor_pt + '/size_liq_all.csv', index_col=0, parse_dates=True)
cap = pd.pivot_table(cap, index=cap.index, columns='code', values=cap.columns[-1])

factors_file = os.listdir(factor_pt)

out = {}
out_rt = {}
hml = {}
for f in factors_file:
    print(f)

    factor = pd.read_csv(factor_pt + '/' + f, index_col=0, parse_dates=True)
    factor = pd.pivot_table(factor, index=factor.index, columns='code', values=factor.columns[-1])

    # TODO 作弊代码 待删
    # if f == 'accural.csv':
    #     factor *= -1

    ft = FactorTest()
    ft.form_port(factor, rte_monthly, n=n, weight=weight, cap=cap)
    ft.group_des()
    ft.factor_decile_description()
    ft.t_test_hml()
    fig = ft.fig_decile_rt()
    # plt.title(f + '_group' + str(n) + '_' + weight)
    plt.savefig(out_pt + '/factor_test/' + f + '_group' + str(n) + '_' + weight + '.png')
    out[f] = ft.hml_t
    out_rt[f] = ft.group_rt_mean
    hml[f] = ft.hml
    # print(ft.group_rt_mean)
    print(ft.hml_t)

# print(out)
hml = pd.DataFrame(hml).T
corr = pd.DataFrame(np.corrcoef(hml), index=hml.index, columns=hml.index)
corr.to_csv(out_pt + '/factor_test/corr_group' + str(n) + '_' + weight + '.csv')

pd.Series(out).to_csv(out_pt + '/factor_test/t_group' + str(n) + '_' + weight + '.csv')
pd.DataFrame(out_rt).to_csv(out_pt + '/factor_test/mean_rt_group' + str(n) + '_' + weight + '.csv')

# TODO culmul_rt 因子可以考虑修改一下，不用月度平均，而是本月最后一个交易日向前滚动20日，作为本月因子
