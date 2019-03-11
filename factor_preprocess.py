# 因子部分的数据预处理，综合基本面/宏观/行业三类因子

from config import config

factor_pt = config.get('PATH', 'factor_pt')
indus_pt = config.get('PATH', 'data_pt') + '/indus/indus.csv'
macro_pt = config.get('PATH', 'data_pt') + '/macro'


