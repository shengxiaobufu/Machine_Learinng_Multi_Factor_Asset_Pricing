# elastic_net模型
import sys
sys.path.append("..")

from give_dataset import X, y
from sklearn.linear_model import ElasticNet
import numpy as np
from common_func import r2_oos, give_rolling_set

alphas = (np.arange(90) + 1)/30
out = []
for a in alphas:
    print('alpha: ', a)
    model = ElasticNet(fit_intercept=True, alpha=a, l1_ratio=a)
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

best_alpha = alphas[np.argmax(out)]
print('Best Alpha on CV set:', best_alpha)

model = ElasticNet(fit_intercept=True, alpha=best_alpha, l1_ratio=best_alpha)
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
