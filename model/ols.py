# OLS模型
import sys
sys.path.append("..")

from give_dataset import X, y
import numpy as np
from common_func import r2_oos, give_rolling_set
from mxnet import init, autograd, nd
from mxnet.gluon import nn, Trainer
from mxnet.gluon import loss as gloss
from mxnet.gluon import data as gdata

# model = LinearRegression(fit_intercept=True)

test_R2 = []

for train_X, train_y, cv_X, cv_y, test_X, test_y in give_rolling_set(X, y, method='rolling'):
    # MODEL ==========================================
    batch_size = 500
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.HuberLoss()
    trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})
    num_epochs = 100
    # ================================================
    features = nd.array(train_X.values)
    label = nd.array(train_y.values)
    dataset = gdata.ArrayDataset(features, label)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        l = loss(net(features), label)
        print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

    # OLS不需要cv
    test_pred_i = net(nd.array(test_X)).asnumpy()
    test_r2_oos = r2_oos(test_y.values, test_pred_i)
    print('Test R2-oos:', test_r2_oos)
    test_R2.append(test_r2_oos)

print('AVE of all R2_OOS:', np.mean(test_R2))


