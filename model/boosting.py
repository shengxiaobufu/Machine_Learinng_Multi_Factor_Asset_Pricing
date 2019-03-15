# boosting: GBRT梯度提升树
# TODO 优化boosting，汇总第一版结果
# TODO 放宽现有因子的参数
# TODO 加入新因子：BARRA，中国文献中提出的显著因子

import sys
sys.path.append("..")

from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from common_func import r2_oos, give_rolling_set
from config import config
import logging

model_log_pt = config.get('PATH', 'log_pt') + '/model.log'

logging.basicConfig(filename=model_log_pt,
                    format='%(asctime)s-%(levelname)s:%(message)s',
                    level=logging.INFO,
                    filemode='a',
                    datefmt="%Y-%m-%d %H:%M:%S")

logging.info(' ')
logging.info('Start OPT ===================================================================================')
logging.info('Model: Boosting')

out_pt = config.get('PATH', 'out_pt')

from give_dataset import X, y
logging.info(' ')

# boosting params
num_trees = (np.arange(8) + 1) * 20 + 30
learning_rate = [0.001, 0.01, 0.03, 0.05, 0.1]
subsample = [0.5, 0.6, 0.7, 0.8]
loss = ['ls', 'huber']

# cart tree params
max_depth = [2, 3, 4, 5, 6]

params = []
for t in num_trees:
    for d in max_depth:
        for l in learning_rate:
            for s in subsample:
                for o in loss:
                    params.append({'num_trees': t, 'max_dep': d, 'lr': l, 'subsample': s, 'loss': o})

out_dict = []
best_r2_oos = []
for train_X, train_y, cv_X, cv_y, test_X, test_y in give_rolling_set(X, y, method='rolling'):
    out_cv = []
    for p in params:
        # print('params: ', p)
        # model = GradientBoostingClassifier(max_depth=p['max_dep'], n_estimators=p['num_trees'],
        #                                    max_features=p['max_fea'], min_samples_split=2)
        model = GradientBoostingRegressor(max_depth=p['max_dep'], n_estimators=p['num_trees'],
                                          learning_rate=p['lr'], max_features=p['max_dep'],
                                          min_samples_split=10, loss=p['loss'], min_samples_leaf=10,
                                          subsample=p['subsample'])
        model.fit(train_X, train_y)
        pred_cv = model.predict(cv_X)
        perfor = r2_oos(cv_y, pred_cv)
        # print(perfor)
        logging.info('params: ' + str(p) + '. CV r2-oos:' + str(perfor))
        p['cv_perfor'] = perfor
        p['cv_year'] = np.max(list(cv_X.index.year))
        print(p)
        out_dict.append(p)
        out_cv.append(perfor)
    best_p = params[np.argmax(out_cv)]
    model = GradientBoostingRegressor()
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    best_perfor = r2_oos(test_y, pred_y)

    best_r2_oos.append(best_perfor)
    msg = 'Best params: ' + str(best_p) + '. R2-oos under best alpha: ' + str(best_perfor)
    print(msg)
    logging.info(msg)
    logging.info(' ')

pd.DataFrame(out_dict).to_csv(out_pt + '/boosting/boosting_4_1_1_rolling.csv', encoding='utf8')

r2 = np.mean(best_r2_oos)
print('Ave R2_OOS within all test set: ' + str(r2))
logging.info('Ave R2_OOS within all test set: ' + str(r2))

logging.info('End OPT ===================================================================================')
logging.info(' ')



