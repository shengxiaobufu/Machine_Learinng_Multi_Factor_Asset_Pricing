# OLS模型
import sys
sys.path.append("..")

from give_dataset import X, y
from sklearn.linear_model import LinearRegression
import numpy as np
from common_func import r2_oos, give_rolling_set


model = LinearRegression(fit_intercept=True)
test_true = []
test_pred = []
for train_X, train_y, cv_X, cv_y, test_X, test_y in give_rolling_set(X, y, method='rolling'):
    model.fit(train_X, train_y)
    # OLS不需要cv
    print('pred rt: ', model.predict(test_X))
    test_true += list(test_y.values)
    test_pred += list(model.predict(test_X))

print('R2_OOS:', r2_oos(np.array(test_true), np.array(test_pred)))






