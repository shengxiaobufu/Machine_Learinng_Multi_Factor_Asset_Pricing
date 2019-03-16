# 汇总参数优化结果，做最后的数据分析：
#   1. 全样本期间 & 16年以前整体R2计算
#   2. 输入变量有效性分析，基于各算法下的最优参数组合

import sys
sys.path.append("..")
from config import config
from give_dataset import X, y
import numpy as np
import pandas as pd
from common_func import r2_oos, give_rolling_set
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from matplotlib import pyplot as plt


out_pt = config.get('PATH', 'out_pt')
factors_name = X.columns
factors_name = [i.replace('_monthly', '').replace('macro-', '') for i in factors_name]


# elastic net：
def elastic_net():
    true_y = []
    pred_y = []
    factor_importance = pd.DataFrame()
    alphas = [0.092, 0.12, 0.128, 0.028, 0.092, 0.088]
    best_r2_oos = []
    count = 0
    for train_X, train_y, cv_X, cv_y, \
        test_X, test_y in give_rolling_set(X, y, train_len=5,
                                           cv_len=1, test_len=1, method='rolling'):
        cut = (np.sum(np.isnan(train_X)) == 0).values
        train_X = train_X.iloc[:, cut]
        cv_X = cv_X.iloc[:, cut]
        test_X = test_X.iloc[:, cut]
        # print(train_X.shape)
        model = ElasticNet(fit_intercept=True, alpha=alphas[count], l1_ratio=alphas[count])

        model.fit(train_X, train_y)
        pred_y_ = model.predict(test_X)
        num_fea_best = np.sum(model.coef_ != 0)
        best_perfor = r2_oos(test_y, pred_y_)
        true_y += list(test_y.values)
        pred_y += list(pred_y_)

        factor_importance_i = pd.Series(model.coef_[model.coef_ != 0],
                                        index=factors_name[model.coef_ != 0], name=count)
        factor_importance_i = np.abs(factor_importance_i)/np.sum(np.abs(factor_importance_i))
        factor_importance = pd.concat([factor_importance, factor_importance_i], axis=1)

        best_r2_oos.append(best_perfor)
        msg = 'Best alpha: ' + str(alphas[count]) + '. R2-oos under best alpha: ' + str(best_perfor) \
              + '. Num of fea choosed: ' + str(num_fea_best)
        print(msg)
        count += 1

        # 只看16年以前的结果
        if count >= 4:
            break

    r2 = np.mean(best_r2_oos)
    print('Ave R2_OOS within all test set: ' + str(r2))
    all_r2 = r2_oos(np.array(true_y), np.array(pred_y))
    print('R2-oos during all test set:', all_r2)

    factor_importance.columns = [2013, 2014, 2015, 2016]
    factor_importance = factor_importance.fillna(0)
    factor_importance['ave'] = np.mean(factor_importance, axis=1)
    factor_importance = factor_importance.sort_values('ave', ascending=False)
    factor_importance.to_csv(out_pt + '/factor_importance_elastic_net.csv')
    factor_importance.index = [i.replace('_monthly', '').replace('macro-', '')
                               for i in factor_importance.index]
    plt.figure(figsize=(10, 8))
    plt.barh(factor_importance.index[:20][::-1], factor_importance['ave'].values[:20][::-1])
    # plt.show()
    plt.tight_layout()
    plt.savefig(out_pt + '/factor_importance_elastic_net.png')


def random_forest():
    true_y = []
    pred_y = []
    factor_importance = pd.DataFrame()

    max_depth = [2, 2, 2, 6, 2, 2]
    max_features = [2, 2, 2, 6, 2, 2]
    params = [{'max_dep': max_depth[i], 'max_fea': max_features[i]} for i in range(len(max_depth))]

    best_r2_oos = []
    count = 0
    for train_X, train_y, cv_X, cv_y, \
        test_X, test_y in give_rolling_set(X, y, train_len=5,
                                           cv_len=1, test_len=1, method='rolling'):
        cut = (np.sum(np.isnan(train_X)) == 0).values
        train_X = train_X.iloc[:, cut]
        test_X = test_X.iloc[:, cut]

        model = RandomForestRegressor(n_estimators=500, max_depth=params[count]['max_dep'],
                                      max_features=params[count]['max_fea'],
                                      min_samples_split=2, random_state=0)

        model.fit(train_X, train_y)
        pred_y_ = model.predict(test_X)
        best_perfor = r2_oos(test_y, pred_y_)
        true_y += list(test_y.values)
        pred_y += list(pred_y_)

        factor_importance_i = pd.Series(model.feature_importances_\
                                        [model.feature_importances_.argsort()[::-1][:20]],
                                        index=np.array(factors_name)[model.feature_importances_.argsort()[::-1][:20]],
                                        name=count)
        factor_importance_i = np.abs(factor_importance_i)/np.sum(np.abs(factor_importance_i))
        factor_importance = pd.concat([factor_importance, factor_importance_i], axis=1)

        best_r2_oos.append(best_perfor)
        msg = 'Best params: ' + str(params[count]) + '. R2-oos under best params: ' + str(best_perfor)
        print(msg)
        count += 1

        # 只看16年以前的结果
        if count >= 4:
            break

    r2 = np.mean(best_r2_oos)
    print('Ave R2_OOS within all test set: ' + str(r2))
    all_r2 = r2_oos(np.array(true_y), np.array(pred_y))
    print('R2-oos during all test set:', all_r2)

    factor_importance.columns = [2013, 2014, 2015, 2016]
    factor_importance = factor_importance.fillna(0)
    factor_importance['ave'] = np.mean(factor_importance, axis=1)
    factor_importance = factor_importance.sort_values('ave', ascending=False)
    factor_importance.to_csv(out_pt + '/factor_importance_rf.csv')

    plt.figure(figsize=(10, 8))
    plt.barh(factor_importance.index[:20][::-1], factor_importance['ave'].values[:20][::-1])
    # plt.show()
    plt.tight_layout()
    plt.savefig(out_pt + '/factor_importance_rf.png')


if __name__ == '__main__':
    random_forest()

