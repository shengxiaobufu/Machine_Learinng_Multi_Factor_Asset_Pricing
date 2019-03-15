# 给出数据，方便统一修改所有模型使用的数据
import sys
sys.path.append("..")

from config import config
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import logging

model_log_pt = config.get('PATH', 'log_pt') + '/model.log'

logging.basicConfig(filename=model_log_pt,
                    format='%(asctime)s-%(levelname)s:%(message)s',
                    level=logging.INFO,
                    filemode='a',
                    datefmt="%Y-%m-%d %H:%M:%S")

#################################### 参数设置 #######################################
data_pt = config.get('PATH', 'data_pt')
print("Now we are use", data_pt + '/data_all.csv')
logging.info("Now we are use " + data_pt + '/data_all.csv')
data_all = pd.read_csv(data_pt + '/data_all.csv', index_col=0, parse_dates=True)
####################################################################################

data_all.columns = ['code'] + list(data_all.columns)[1:]
y = data_all['rte']

indus_col = ['indus']
macor_col = [i for i in data_all.columns if i.split('-')[0] == 'macro']
assert len(macor_col) == 8

factor_col = [i for i in data_all.columns if i.split('-')[0] != 'macro'
              and i not in ['indus', 'code', 'rte']]
assert len(factor_col) == 14

# 行业因子one-hot
one_hot = OneHotEncoder()
indus = one_hot.fit_transform(data_all[indus_col]).toarray()
indus_code = [str(i) for i in one_hot.active_features_]
assert indus.shape[1] == len(indus_code)


# 在这里不做标准化，因为整体标准化有用到未来数据的嫌疑
factor_scaled = data_all[factor_col].values
macro_scaled = data_all[macor_col].values

# 计算交叉项
f_x_m_all = None
names = []
f_num = factor_scaled.shape[1]
for i in range(macro_scaled.shape[1]):
    macro_i = macro_scaled[:, i]
    macro_i_name = macor_col[i]
    f_x_m_i = factor_scaled * np.repeat(macro_i, f_num).reshape(-1, f_num)
    names += [i + ' * ' + macro_i_name for i in factor_col]
    if f_x_m_all is None:
        f_x_m_all = f_x_m_i
    else:
        f_x_m_all = np.concatenate([f_x_m_all, f_x_m_i], axis=1)

assert f_x_m_all.shape[1] == 14*8
assert len(names) == 14*8

assert factor_scaled.shape[0] == f_x_m_all.shape[0]
assert factor_scaled.shape[0] == indus.shape[0]
assert factor_scaled.shape[0] == y.shape[0]


X = pd.DataFrame(np.concatenate([factor_scaled, f_x_m_all, indus], axis=1),
                 index=data_all.index, columns=factor_col + names + indus_code)
print(X.shape)
print(y.shape)


