# 给出数据，方便统一修改所有模型使用的数据

import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class FactorTest(object):
    def __init__(self):
        self.port_rt = None
        self.port_stock = None
        self.port_cap = None
        self.n = None
        return

    def form_port(self, factor, rt, n, cap=None, weight='cap'):
        assert weight in ['ave', 'cap'], 'unknown weight method.'
        if weight=='cap':
            assert cap is not  None, 'please give cap if you want to use cap weight.'
        else:
            cap = pd.DataFrame(np.ones(factor.shape), index=factor.index, columns=factor.columns)
        self.n = n
        # factor和cap需要滞后一期（观察上月，持有本月）
        factor = pd.DataFrame(factor.values[:-1, :], index=factor.index[1:], columns=factor.columns)
        cap = pd.DataFrame(cap.values[:-1, :], index=cap.index[1:], columns=cap.columns)

        # 格式标准化
        rt.columns = [int(i) for i in rt.columns]
        factor.columns = [int(i) for i in factor.columns]
        cap.columns = [int(i) for i in cap.columns]

        # 统一股票池和时间区间
        stocks = list(set(list(rt.columns)) & set(list(factor.columns)) & set(list(cap.columns)))
        times = sorted(list(set(list(rt.index)) & set(list(factor.index)) & set(list(cap.index))))
        factor = factor[stocks].T[times].T
        rt = rt[stocks].T[times].T
        cap = cap[stocks].T[times].T
        assert list(rt.columns) == list(cap.columns) and list(rt.columns) == list(factor.columns)
        assert list(rt.index) == list(cap.index) and list(rt.index) == list(factor.index)
        rt = rt.values
        cap = cap.values

        # 排序分组
        sort = factor.values.argsort()
        stocks = np.array(factor.columns)
        times = np.array(factor.index)
        factor = factor.values
        # print(np.sum(np.sum(np.isnan(factor))))
        nans = (np.isnan(factor) + np.isnan(rt) + np.isnan(cap)) > 0
        # print(nans)
        factor[nans] = np.nan
        # print(np.sum(np.sum(np.isnan(factor))))
        self.port_factor = {p: {} for p in range(n)}
        self.port_rt = {p: {} for p in range(n)}
        self.port_stock = {p: {} for p in range(n)}
        self.port_cap = {p: {} for p in range(n)}
        self.port_rt_weighted = {p: {} for p in range(n)}
        for i in range(len(sort)):
            s = sort[i]
            t = times[i]
            stock_num_i = len(factor[i, :][~np.isnan(factor[i, :])])
            len_p = stock_num_i//n
            final_len_p = len_p + stock_num_i % n
            for p in range(n-1):
                sp = s[p * len_p : (p+1) * len_p]
                rt_p = rt[i, sp]
                stock_p = stocks[sp]
                cap_p = cap[i, sp]
                cap_p[np.isnan(cap_p)] = 0
                factor_p = factor[i, sp]
                self.port_factor[p][t] = factor_p
                self.port_rt_weighted[p][t] = np.nansum(rt_p * cap_p / np.sum(cap_p))
                self.port_rt[p][t] = rt_p
                self.port_stock[p][t] = stock_p
                self.port_cap[p][t] = cap_p

            sn = s[len_p * (n-1): len_p * (n-1) + final_len_p]
            rt_n = rt[i, sn]
            stock_n = stocks[sn]
            cap_n = cap[i, sn]
            cap_n[np.isnan(cap_n)] = 0
            factor_n = factor[i, sn]
            self.port_factor[n-1][t] = factor_n
            self.port_rt[n-1][t] = rt_n
            self.port_stock[n-1][t] = stock_n
            self.port_cap[n-1][t] = cap_n
            self.port_rt_weighted[n-1][t] = np.nansum(rt_n * cap_n / np.sum(cap_n))
        return

    def group_des(self):
        self.port_rt_weighted = pd.DataFrame(self.port_rt_weighted)
        self.group_rt_mean = np.mean(self.port_rt_weighted, axis=0)

    def factor_decile_description(self):
        self.mean_f = []
        for d in self.port_factor.keys():
            # print(d)
            dict_ = self.port_factor[d]
            mean_dict = [np.mean(dict_[p]) for p in dict_]
            # print(mean_dict)
            self.mean_f.append(mean_dict)
        self.mean_f = pd.DataFrame(self.mean_f)

    def t_test_hml(self):
        self.hml = self.port_rt_weighted[self.n-1] - self.port_rt_weighted[1]
        self.hml_t = ttest_1samp(self.hml, 0)[0]

    def fig_decile_rt(self):
        culmul_rt = np.cumprod(self.port_rt_weighted + 1)
        fig = plt.figure(figsize=(14, 8))
        plt.plot(culmul_rt)
        plt.legend(np.arange(self.n))
        # fig.show()
        return fig

