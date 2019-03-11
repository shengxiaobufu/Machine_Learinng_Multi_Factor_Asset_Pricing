# 计算月度的行业因子，基于CSMAR数据库给出的证监会行业分类的一级分类（19类）

from config import config
import pandas as pd
from datetime import date
from data_deal_func import fullfill_monthly

data_pt = config.get('PATH', 'data_pt')

indus = pd.read_csv(data_pt + '/indus.csv', index_col=1, parse_dates=True, encoding='gb2312')
indus_codes_sec = list(set(indus.iloc[:, 2]))
indus_codes_fir = list(set([i[0] for i in indus.iloc[:, 2]]))
print('一级分类数量：', len(indus_codes_fir))
print('二级分类数量：', len(indus_codes_sec))

count = 0
indus_dir = {}
for c in indus_codes_fir:
    indus_dir[c] = count
    count += 1

indus['indus_1st'] = [indus_dir[i[0]] for i in indus.iloc[:, 2]]
time_indus = pd.pivot_table(indus, values='indus_1st', index=indus.index, columns='Symbol')
time_indus.index = [date(i.year, i.month, 1) for i in time_indus.index]

time_indus = fullfill_monthly(2005, 2018, time_indus)
final_indus_code = list(time_indus.values.reshape(-1, 1).T)
all_code = []
all_date = []
for i in time_indus.columns:
    # time_indus[i][~np.isnan(time_indus[i])].values
    all_code += [i] * len(time_indus)
    all_date += list(time_indus.index)

final_indus_code = pd.DataFrame([all_code, final_indus_code], index=['code', 'indus'], columns=all_date).T
final_indus_code.to_csv(data_pt + '/factors/indus.csv', encoding='utf8')

with open(data_pt + '/factors/indus_dir.txt', 'w') as f:
    for k in indus_dir:
        f.write(k + ': ' + str(indus_dir[k]))
        f.write('/n')

