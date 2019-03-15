# 所有模型通用的一些方法
import numpy as np
import math
from datetime import date
from sklearn.preprocessing import scale
import pandas as pd
import logging
from config import config

model_log_pt = config.get('PATH', 'log_pt') + '/model.log'

logging.basicConfig(filename=model_log_pt,
                    format='%(asctime)s-%(levelname)s:%(message)s',
                    level=logging.INFO,
                    filemode='a',
                    datefmt="%Y-%m-%d %H:%M:%S")

def r2_oos(true, pred):
    """
    模型的评价标准：Out-of-sample R2
    """
    return 1 - np.sum((true - pred) ** 2) / np.sum(true ** 2)


def diebold_mariano_test():
    # TODO
    return


def give_rolling_set(X, y, train_len=4, cv_len=1, test_len=1, method='rolling', allow_nan=True):
    assert method in ['rolling', 'recursive'], 'Unknown method.'
    assert np.sum(np.isnan(y)) == 0, 'y contains NAN, please check data.'
    if not allow_nan:
        assert np.sum(np.sum(np.isnan(X))) == 0, 'Input contains NAN, allow_nan to set nan as median, or check data.'

    year_all = sorted(list(set(X.index.year)))
    length = len(year_all)
    rolling_time = math.ceil((length - (train_len + cv_len))/test_len)

    for i in range(rolling_time):
        if method == 'rolling':
            train_start = year_all[test_len * i]
            train_end = train_start + train_len
            cv_end = train_end + cv_len
            test_end = cv_end + test_len
        else:
            train_start = year_all[test_len * i]
            train_end = train_start + train_len
            cv_end = train_end + cv_len
            test_end = cv_end + test_len
            train_start = year_all[0]
        msg = 'Train: ' + str(train_start) + ' to ' + str(train_end-1) \
              + '. CV: ' + str(train_end) + ' to ' + str(cv_end-1) \
              + '. Test: ' + str(cv_end) + ' to ' + str(test_end-1)
        print(msg)
        logging.info(msg)

        train_X = scale_from_all_his(X, train_start, train_end, allow_nan=allow_nan)
        train_y = y[(y.index.year >= train_start) * (y.index.year < train_end)]
        cv_X = scale_from_all_his(X, train_end, cv_end, allow_nan=allow_nan)
        cv_y = y[(y.index.year >= train_end) * (y.index.year < cv_end)]
        test_X = scale_from_all_his(X, cv_end, test_end, allow_nan=allow_nan)
        test_y = y[(y.index.year >= cv_end) * (y.index.year < test_end)]

        yield train_X, train_y, cv_X, cv_y, test_X, test_y


def add_month_single(date_):
    year = date_.year
    month = date_.month + 1
    if date_.month == 12:
        year += 1
        month -= 12
    return date(year, month, 1)


def add_month(dates):
    dates = [i.date() for i in dates]
    return [add_month_single(d) for d in dates]


def scale_from_all_his(all_data, start_year, end_year, tail_thres=5, allow_nan=True):
    # 标准化
    scaled_to_now = pd.DataFrame(scale(all_data[all_data.index.year < end_year]),
                                 index=all_data.index[all_data.index.year < end_year],
                                 columns=all_data.columns)
    scaled_cuted = scaled_to_now[scaled_to_now.index.year >= start_year]
    # 去极值
    scaled_cuted[scaled_cuted > tail_thres] = tail_thres
    scaled_cuted[scaled_cuted < -tail_thres] = -tail_thres
    # 填空值（allow_nan = False 意味着不允许空值，直接报错返回）
    if allow_nan:
        nans_col = scaled_cuted.iloc[:, (np.sum(np.isnan(scaled_cuted)) > 0).values].columns
        for c in nans_col:
            fill = np.nanmedian(scaled_cuted[c])
            scaled_cuted[c] = scaled_cuted[c].fillna(fill)
    return scaled_cuted

