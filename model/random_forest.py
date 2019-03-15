# RF模型
import sys
sys.path.append("..")

from sklearn.ensemble import RandomForestRegressor
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
    logging.info('Model: Random Forest')

    from give_dataset import X, y
    logging.info(' ')

    out_pt = config.get('PATH', 'out_pt')

    max_depth = np.arange(4) + 2
    max_features = np.arange(4) + 2
    params = [{'max_dep': i, 'max_fea': j} for i in max_depth for j in max_features]

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
        # print(np.sum(np.sum(np.isnan(cv_X))))
        for p in params:
            # print(p)
            model = RandomForestRegressor(n_estimators=500, max_depth=p['max_dep'], max_features=p['max_fea'],
                                          min_samples_split=2, random_state=0)
            model.fit(train_X, train_y)
            pred_cv = model.predict(cv_X)
            perfor = r2_oos(cv_y, pred_cv)
            print('params: ' + str(p) + '. CV r2-oos:' + str(perfor))
            logging.info('params: ' + str(p) + '. CV r2-oos:' + str(perfor))
            out_cv.append(perfor)

        best_p = params[np.argmax(out_cv)]
        model = RandomForestRegressor(n_estimators=1000, max_depth=best_p['max_dep'], max_features=best_p['max_fea'],
                                      min_samples_split=2, random_state=0)
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        best_perfor = r2_oos(test_y, pred_y)

        best_r2_oos.append(best_perfor)
        msg = 'Best params: ' + str(best_p) + '. R2-oos under best alpha: ' + str(best_perfor)
        print(msg)
        logging.info(msg)
        logging.info(' ')

    r2 = np.mean(best_r2_oos)
    print('Ave R2_OOS within all test set: ' + str(r2))
    logging.info('Ave R2_OOS within all test set: ' + str(r2))

    logging.info('End OPT ===================================================================================')
    logging.info(' ')

