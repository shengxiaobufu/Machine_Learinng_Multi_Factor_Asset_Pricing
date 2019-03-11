import pandas as pd
import numpy as np
from datetime import date


def fullfill_monthly(start_year, end_year, data_):
    year = np.arange(end_year-start_year+1) + start_year
    month = np.arange(12) + 1
    all_time = []
    for y in year:
        for m in month:
            all_time.append(date(y, m, 1))

    all_time_df = pd.DataFrame([], index=all_time)
    data_ = pd.concat([all_time_df, data_], axis=1).fillna(method='pad')
    return data_


def daily2monthly(daily_data, group_c=['year_month', 'code']):
    daily_data['year_month'] = [date.strftime(date(i.year, i.month, 1), '%Y-%m')
                                for i in daily_data.index]
    group = daily_data.groupby(group_c)
    mean_cap = group.mean()
    return mean_cap
