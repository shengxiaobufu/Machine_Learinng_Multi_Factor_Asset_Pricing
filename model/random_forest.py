# RF模型
import sys
sys.path.append("..")

from give_dataset import X, y
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from common_func import r2_oos, give_rolling_set

num_trees = (np.arange(20) + 1) * 10
max_depth = np.arange(10) + 1
max_features = (np.arange(5) + 1) * 2

params = []
for i in num_trees:
    for j in max_depth:
        for k in max_features:
            params.append({'num_trees': i, 'max_dep': j, 'max_fea': k})

out = []
for p in params:
    print('params: ', p)
    model = RandomForestRegressor(max_depth=p['max_dep'], n_estimators=p['num_trees'],
                                  max_features=p['max_fea'])
    cv_true = []
    cv_pred = []
    for train_X, train_y, cv_X, cv_y, test_X, test_y in give_rolling_set(X, y, method='rolling'):
        model.fit(train_X, train_y)
        # OLS不需要cv
        # print('pred rt: ', model.predict(test_X))
        cv_true += list(test_y.values)
        cv_pred += list(model.predict(test_X))
    r2 = r2_oos(np.array(cv_true), np.array(cv_pred))
    out.append(r2)
    print('R2_OOS:', r2)

best_p = params[np.argmax(out)]
print('Best Alpha on CV set:', best_p)

print('---------------------FINAL TEST PERFORMANCE-------------------')
model = RandomForestRegressor(max_depth=best_p['max_dep'], n_estimators=best_p['num_trees'],
                              max_features=best_p['max_fea'])
test_true = []
test_pred = []
for train_X, train_y, cv_X, cv_y, test_X, test_y in give_rolling_set(X, y, method='rolling'):
    model.fit(train_X, train_y)
    # OLS不需要cv
    # print('pred rt: ', model.predict(test_X))
    test_true += list(test_y.values)
    test_pred += list(model.predict(test_X))
r2 = r2_oos(np.array(test_true), np.array(test_pred))
out.append(r2)
print('R2_OOS:', r2)
