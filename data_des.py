# 数据的描述性统计

from config import config
import pandas as pd
import numpy as np
import os

data_pt = config.get('PATH', 'data_pt')

data_all = pd.read_csv(data_pt + '/data_all.csv', index_col=0, parse_dates=True)
data_all.columns = ['code'] + list(data_all.columns)[1:]

years = np.arange(12) + 2007
lens_y = []
stock_num_y = []
for y in years:
    data_y = data_all[data_all.index.year == y]
    lens_y.append(len(data_y))
    stock_num_y.append(len(list(set(data_y['code']))))


# DEMO DATA
demo_data = data_all[data_all.index.year >= 2010]
demo_data = demo_data[demo_data['code'] < 20]
print(demo_data.shape)
demo_data.to_csv(data_pt + '/demo_data.csv', encoding='utf8')
