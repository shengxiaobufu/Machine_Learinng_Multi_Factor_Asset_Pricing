# elastic_net模型
# 目前alpha从0.26-3都是一样的R2，-0.027648

import sys
sys.path.append("..")

from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
from common_func import r2_oos, give_rolling_set
from config import config
import logging
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_len', default=2)
    parser.add_argument('--cv_len', default=1)
    parser.add_argument('--test_len', default=1)
    parser.add_argument('--rolling_method', default='rolling')
    args = parser.parse_args()
    main_params = vars(args)

    model_log_pt = config.get('PATH', 'log_pt') + '/model.log'

    logging.basicConfig(filename=model_log_pt,
                        format='%(asctime)s-%(levelname)s:%(message)s',
                        level=logging.INFO,
                        filemode='a',
                        datefmt="%Y-%m-%d %H:%M:%S")

    logging.info(' ')
    logging.info('Start OPT ===================================================================================')
    logging.info('Model: Elastic Net')

    from give_dataset import X, y
    logging.info(' ')
    out_pt = config.get('PATH', 'out_pt')

    alphas = np.linspace(0, 0.2, 51)[1:]
    best_r2_oos = []
    for train_X, train_y, cv_X, cv_y, \
        test_X, test_y in give_rolling_set(X, y, train_len=int(main_params['train_len']),
                                           cv_len=int(main_params['cv_len']), test_len=int(main_params['test_len']),
                                           method=main_params['rolling_method']):
        out_cv = []
        cut = (np.sum(np.isnan(train_X)) == 0).values
        train_X = train_X.iloc[:, cut]
        cv_X = cv_X.iloc[:, cut]
        test_X = test_X.iloc[:, cut]
        print(train_X.shape)
        aa = np.sum(np.isnan(cv_X))
        for a in alphas:
            model = ElasticNet(fit_intercept=True, alpha=a, l1_ratio=a)
            model.fit(train_X, train_y)
            pred_cv = model.predict(cv_X)
            perfor = r2_oos(cv_y, pred_cv)
            num_fea = np.sum(model.coef_ != 0)
            logging.info('alpha & l1: ' + str(a) + '. Num of features selected: ' + str(num_fea)
                         + '. CV r2-oos:' + str(perfor))
            out_cv.append(perfor)
            if num_fea < 3:     # 限制自变量选择，要求至少选出三个来
                break

        best_a = alphas[np.argmax(out_cv)]
        model = ElasticNet(fit_intercept=True, alpha=best_a, l1_ratio=best_a)
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        num_fea_best = np.sum(model.coef_ != 0)
        best_perfor = r2_oos(test_y, pred_y)

        best_r2_oos.append(best_perfor)
        msg = 'Best alpha: ' + str(best_a) + '. R2-oos under best alpha: ' + str(best_perfor) \
              + '. Num of fea choosed: ' + str(num_fea_best)
        print(msg)
        logging.info(msg)
        logging.info(' ')

    r2 = np.mean(best_r2_oos)
    print('Ave R2_OOS within all test set: ' + str(r2))
    logging.info('Ave R2_OOS within all test set: ' + str(r2))

    logging.info('End OPT ===================================================================================')
    logging.info(' ')
