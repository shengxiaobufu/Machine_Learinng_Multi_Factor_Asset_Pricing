# 因子计算: 考虑到内存限制和insert方便，先导出csv到data/factors文件夹，再统一传入
# 所有因子均以日度为基础，然后去月内平均，作为当月的月度因子，用来预测下月收益率
# 为了内存友好，封装成函数，可以共用数据的封装在一起
# 在此不做归一化

from config import config
import pandas as pd
import numpy as np
from datetime import date
from db_related import select_data


def minus_year(dates):
    dates_ = []
    for d in dates:
        year = d.year - 1
        month = d.month
        if d.month == 2 and d.day == 29:
            day = 28
        else:
            day = d.day
        dates_.append(date(year, month, day))
    return dates_


def vs_last_year(this_year_t, column='A001000000', out_column='inves', type_='growth_rate'):
    """
    和去年同期进行比较
    :param this_year_t: 当年数据，必须包含code，date字段，以及需要和去年同期比较的字段
    :param column: 比较字段的字段名
    :param out_column: 输出结果的字段名
    :param type_: 输出结果类型，rate代表增长率，amount代表增长额
    :return: df格式输出，日期索引，内容为两列，一列code，一列输出的增长率或增长额
    """
    assert type_ in ['growth_rate', 'growth_amount'], 'Unknown type_!'
    stocks = list(set(this_year_t['code']))
    out = []
    code = []
    index = []
    for stock in stocks:
        print(stock)
        this_t = this_year_t[this_year_t['code'] == stock].sort_index()
        raw_t = list(this_t.index)
        index += raw_t
        raw_t = np.array(raw_t)
        code += [stock for i in range(len(raw_t))]
        last_t_temp = minus_year(raw_t)
        last_t = []
        start = raw_t[0]
        for t in last_t_temp:
            if t < start:
                # print(t, 'skip')
                out.append(np.nan)
                continue
            if t not in raw_t:
                # print(t, 'not in raw t, find nearest')
                last_t.append(raw_t[np.argmin(np.abs(raw_t - t))])
                continue
            # print(t, 'in raw t')
            last_t.append(t)
        out += list(this_t[column].loc[last_t].values)
    if type_ == 'growth_rate':
        growth = list(this_year_t[column].values / np.array(out) - 1)
    else:
        growth = list(this_year_t[column].values - np.array(out))
    return pd.DataFrame([code, growth], index=['code', out_column], columns=index).T


def daily2monthly(daily_data):
    daily_data['year_month'] = [date.strftime(date(i.year, i.month, 1), '%Y-%m')
                                for i in daily_data.index]
    group = daily_data.groupby(['year_month', 'code'])
    mean_cap = group.mean()
    return mean_cap


def to_rolling_n(n, data_):
    """
    把原始的数据转换成rolling—n-window的形式
    :param n: 滚动n天
    :param data_: 待计算数据
    :return: 整理好的数据，形状为(len(data)-n+1, n)
    """
    window_num = len(data_) - n + 1
    slices = [np.arange(n) + i for i in range(window_num)]
    windowed_data = np.array([data_[i] for i in slices])
    return windowed_data


def rolling_func(n, data_all, func, rolling_column, out_name):
    """
    按照给出的func计算data_滚动n日的rolling结果
    :param n: 滚动n天
    :param data_all: 待计算数据
    :param func: 滚动计算函数
    :param rolling_column: 滚动列列名
    :return: 滚动结果，长度与data_一致，不足n天的部分为nan
    """
    stocks = list(set(data_all['code']))
    index = []
    code = []
    out_all = []
    for stock in stocks:
        data_raw = data_all[rolling_column][data_all['code'] == stock].sort_index()
        data_ = data_raw.values
        if len(data_) < n:
            continue
        windowed_data = to_rolling_n(n, data_)
        out_data = list(np.repeat(np.nan, n-1)) + list(func(windowed_data, axis=1))
        out_all += out_data
        index += list(data_raw.index)
        code += [stock for i in range(len(out_data))]
    out_all_df = pd.DataFrame([code, out_all], columns=index, index=['code', out_name]).T
    return out_all_df


# 1. size: 总市值 & 流通市值
# col 1-2：月底市值  col 3-4：月内平均市值
def get_size():
    select_sql_monthly = 'SELECT `date`, `code`, `cap_liq`, `cap_all` FROM `market_rt_monthly`' \
                         ' WHERE `code` != 0'
    select_sql_daily = 'SELECT `date`, `code`, `cap_liq`, `cap_all` FROM `market_rt_daily` WHERE `code` != 0'
    daily_data = select_data(select_sql_daily).reset_index().set_index('date').iloc[:, 1:]
    mean_cap = daily2monthly(daily_data)
    del daily_data
    raw_data = select_data(select_sql_monthly)
    raw_data['date'] = [date.strftime(date(i.year, i.month, 1), '%Y-%m')
                      for i in raw_data['date']]
    raw_data = raw_data.reset_index().set_index(['date', 'code']).iloc[:, 1:]
    raw_data.index.names = ['year_month', 'code']
    cap_all = pd.merge(raw_data, mean_cap, left_index=True, right_index=True, how='outer')
    cap_all.to_csv(factor_path + '/size_liq_all.csv')
    return


# 2. Value: 市盈率EP/市净率BM/市现率CP —— 当月所有交易日的平均值作为月度因子
#           市盈率：静态市盈率（最新EARNINGS（净利润）/当天总市值）
#           市净率：普通股权益总额(所有者权益总计 - 优先股 - 永续债)/当天总市值
#           市现率：经营活动产生的现金流量净额/当天总市值

def get_value(factor='eq'):
    select_sql_cap = 'SELECT `date`, `code`, `cap_all` FROM `market_rt_daily` WHERE `code` != 0'
    caps = select_data(select_sql_cap).reset_index().set_index(['date', 'code']).iloc[:, 1:]
    if factor == 'ep' or factor == 'all':
        # EP
        select_sql_earnings = 'SELECT `date`, `code`, `B002000000` FROM `report_profit_daily` WHERE `code` != 0'
        earnings = select_data(select_sql_earnings).reset_index().set_index(['date', 'code']).iloc[:, 1:]

        print('get e and p')
        e_p = pd.merge(earnings, caps, left_index=True, right_index=True, how='outer')
        e_p['ep_daily'] = e_p.values[:, 0] / e_p.values[:, 1]
        print('get daily ep')
        ep_monthly = daily2monthly(e_p.iloc[:, 2].reset_index().set_index('date'))
        print('get monthly ep')
        ep_monthly.columns = ['ep_monthly']
        ep_monthly.to_csv(factor_path + '/ep.csv')
        del earnings
        del e_p
        del ep_monthly

    if factor == 'bm' or factor == 'all':
        # BM
        select_sql_book = 'SELECT `date`, `code`, `A003000000` - `A003112101` - `A003112201` ' \
                          'FROM `report_bs_daily` WHERE `code` != 0'
        book_values = select_data(select_sql_book).reset_index().set_index(['date', 'code']).iloc[:, 1:]
        print('get b and m')
        b_m = pd.merge(book_values, caps, left_index=True, right_index=True, how='outer')
        b_m['bm_daily'] = b_m.values[:, 0] / b_m.values[:, 1]
        print('get daily bm')
        bm_monthly = daily2monthly(b_m.iloc[:, 2].reset_index().set_index('date'))
        bm_monthly.columns = ['bm_monthly']
        bm_monthly.to_csv(factor_path + '/bm.csv')
        del book_values
        del b_m
        del bm_monthly

    if factor == 'cp' or factor == 'all':
        # CP
        select_sql_cash = 'SELECT `date`, `code`, `D000100000` ' \
                          'FROM `report_cf_indirect_daily` WHERE `code` != 0'
        cash = select_data(select_sql_cash).reset_index().set_index(['date', 'code']).iloc[:, 1:]
        print('get c and p')
        c_p = pd.merge(cash, caps, left_index=True, right_index=True, how='outer')
        c_p['cp_daily'] = c_p.values[:, 0] / c_p.values[:, 1]
        print('get daily cp')
        cp_monthly = daily2monthly(c_p.iloc[:, 2].reset_index().set_index('date'))
        cp_monthly.columns = ['cp_monthly']
        print('get monthly cp')
        print(cp_monthly.iloc[:10, :])
        cp_monthly.to_csv(factor_path + '/cp.csv')
        # no need to merge
    return


# 3. Profitability: ROE(净利润/所有者权益总计)
def get_profitability():
    select_sql_earnings = 'SELECT `date`, `code`, `B002000000` FROM `report_profit_daily` WHERE `code` != 0'
    select_sql_equity = 'SELECT `date`, `code`, `A003000000` FROM `report_bs_daily` WHERE `code` != 0'
    print('Start calculate profitability')
    earnings = select_data(select_sql_earnings).reset_index().set_index(['date', 'code']).iloc[:, 1:]
    equitys = select_data(select_sql_equity).reset_index().set_index(['date', 'code']).iloc[:, 1:]
    print('get earinings and equitys')
    pro_eq = pd.merge(earnings, equitys, left_index=True, right_index=True, how='outer')
    pro_eq['roe_daily'] = pro_eq.values[:, 0] / pro_eq.values[:, 1]
    print('get daily roe')
    roe_monthly = daily2monthly(pro_eq.iloc[:, 2].reset_index().set_index('date'))
    roe_monthly.columns = ['roe_monthly']
    roe_monthly.to_csv(factor_path + '/roe.csv')
    print('get monthly roe, saved at', factor_path + '/roe.csv')
    return


# 价量相关：
# 4. Volatility: vol_rolling/max
#                vol_rolling: std(rt last 20)
#                max: max(rt last 20)
# 7. Liquidity: abs(rt_daily) / volume_daily
# 9. Reversal: culmul(rt_daily_20days)
def get_price_volume_related(factor):
    # select_sql_rt_volume = 'SELECT `date`, `code`, `rt`, `volume` FROM `market_rt_daily` ' \
    #                        'WHERE `code` <= 10 and code != 0'    # demo data
    select_sql_rt_volume = 'SELECT `date`, `code`, `rt`, `volume` FROM `market_rt_daily` WHERE `code` != 0'
    rt_vol = select_data(select_sql_rt_volume).reset_index().set_index(['date', 'code']).iloc[:, 1:]
    print('get rt and volume')
    # print(rt_vol.columns)

    if factor in ['liq', 'all']:
        liq_daily = pd.DataFrame(np.abs(rt_vol.iloc[:, 0].values)/rt_vol.iloc[:, 1].values,
                                 index=rt_vol.index, columns=['liq_daily'])
        print('get liquidity daily')
        liq_monthly = daily2monthly(liq_daily.reset_index().set_index('date'))
        liq_monthly.columns = ['liq_monthly']
        liq_monthly.to_csv(factor_path + '/liq.csv')
        print('get monthly liq, saved at', factor_path + '/liq.csv')
        del liq_daily
        del liq_monthly

    if factor in ['vol_rolling', 'all']:
        vol_rolling_daily = rolling_func(20, rt_vol.reset_index().set_index('date'),
                                         np.std, 'rt', 'vol_rolling_20_daily')
        print('get vol daily')
        vol_rolling_monthly = daily2monthly(vol_rolling_daily)
        vol_rolling_monthly.columns = ['vol_rolling_20_monthly']
        vol_rolling_monthly.to_csv(factor_path + '/vol_rolling_20.csv')
        print('get monthly vol, saved at', factor_path + '/vol_rolling_20.csv')
        del vol_rolling_daily
        del vol_rolling_monthly

    if factor in ['max', 'all']:
        max_rt_20_daily = rolling_func(20, rt_vol.reset_index().set_index('date'),
                                       np.max, 'rt', 'max_rt_20_daily')
        print('get max daily')
        max_rt_20_monthly = daily2monthly(max_rt_20_daily)
        max_rt_20_monthly.columns = ['max_rt_20_monthly']
        max_rt_20_monthly.to_csv(factor_path + '/max_rt_20.csv')
        print('get monthly max, saved at', factor_path + '/max_rt_20.csv')
        del max_rt_20_daily
        del max_rt_20_monthly

    if factor in ['Reversal', 'all']:
        cumul_rt_data = rt_vol.reset_index().set_index('date')[['code', 'rt']]
        cumul_rt_data['rt'] += 1
        cumul_rt_20_daily = rolling_func(20, cumul_rt_data, np.prod, 'rt', 'cumul_rt_20_daily') - 1
        # 这里要减一！
        print('get Reversal daily')
        cumul_rt_20_monthly = daily2monthly(cumul_rt_20_daily)
        cumul_rt_20_monthly.columns = ['cumul_rt_20_monthly']
        cumul_rt_20_monthly.to_csv(factor_path + '/cumul_rt_20.csv')
        print('get monthly reversal, saved at', factor_path + '/cumul_rt_20.csv')
    return


# 纯财报相关
# 5. Investment：asset_t / asset_t_last_year - 1     去年同一时点的总资产
# 6. Accruals: 应计收入/净营运资产比率
#       应计收入: (流动资产变动 - 货币资金变动) - (流动负债变动 - 短期借款变动 - 应交税费变动) - 折旧摊销
#                折旧摊销 = 固定资产折旧 + 无形资产摊销 + 长期待摊费用摊销
#       Net-operating-assets, NOA： (流动资产 - 流动负债) / 总资产
def get_bs_related(factors):
    # select_sql_assets = 'SELECT `date`, `code`, `A001000000`, `A001100000` - `A002100000` FROM `report_bs_daily` ' \
    #                     'WHERE `code` <= 10 and code != 0'    # demo data
    select_sql_assets = 'SELECT `date`, `code`, `A001000000`, `A001100000` - `A002100000` ' \
                        'FROM `report_bs_daily` WHERE `code` != 0'
    assets_all = select_data(select_sql_assets).reset_index().set_index('date').iloc[:, 1:]
    assets_all.index = [i.date() for i in assets_all.index]
    print('get assets')
    if factors in ['inves', 'all']:
        inves_daily = vs_last_year(assets_all[['code', 'A001000000']])
        print('get daily inves')
        inves_monthly = daily2monthly(inves_daily)
        inves_monthly.columns = ['inves_monthly']
        inves_monthly.to_csv(factor_path + '/inves.csv')
        print('get monthly inves, saved at', factor_path + '/inves.csv')
        del inves_daily
        del inves_monthly
    if factors in ['noa', 'all']:
        assets_all['noa_daily'] = assets_all.iloc[:, 1].values / assets_all.iloc[:, 0].values
        print('get daily noa')
        noa_monthly = daily2monthly(assets_all[['code', 'noa_daily']])
        noa_monthly.columns = ['noa_monthly']
        noa_monthly.to_csv(factor_path + '/noa.csv')
        print('get monthly noa, saved at', factor_path + '/noa.csv')
    return


def get_accruals_1():
    # select_sql_bs = 'SELECT `date`, `code`, ' \
    #                 '`A001100000` - `A001101000` - `A002100000` - `A002101000` - `A002113000` ' \
    #                 'as `cl_minus_std_tp` ' \
    #                 'FROM `report_bs_daily` WHERE `code` <= 10 and code != 0'
    # select_sql_cf = 'SELECT `date`, `code`, `D000103000` + `D000104000` + `D000105000` as dep ' \
    #                 'FROM `report_cf_indirect_daily` WHERE `code` <= 10 and code != 0'      # demo data
    select_sql_bs = 'SELECT `date`, `code`, ' \
                    '`A001100000` - `A001101000` - `A002100000` - `A002101000` - `A002113000` ' \
                    'as `cl_minus_std_tp` ' \
                    'FROM `report_bs_daily` WHERE `code` != 0'
    select_sql_cf = 'SELECT `date`, `code`, `D000103000` + `D000104000` + `D000105000` as dep ' \
                    'FROM `report_cf_indirect_daily` WHERE `code` != 0'
    bs_related = select_data(select_sql_bs).reset_index().set_index('date').iloc[:, 1:]
    dep = select_data(select_sql_cf).reset_index().set_index(['date']).iloc[:, 1:]
    dep.index = [i.date() for i in dep.index]
    bs_related.index = [i.date() for i in bs_related.index]
    print('get accural data')
    bs_related_growth = vs_last_year(bs_related,  column='cl_minus_std_tp', out_column='bs_rela_growth',
                                     type_='growth_amount')
    bs_related_growth = bs_related_growth.reset_index().set_index(['index', 'code'])
    dep = dep.reset_index().set_index(['index', 'code'])
    b_d = pd.merge(bs_related_growth, dep, left_index=True, right_index=True, how='outer')
    b_d['accural_daily'] = b_d.values[:, 0] - b_d.values[:, 1]
    print('get daily ep')
    accural_monthly = daily2monthly(b_d['accural_daily'].reset_index().set_index('index'))
    # TODO 这里的accural需不需要用总资产或者市值做一下横截面的标准化？
    accural_monthly.columns = ['accural_monthly']
    accural_monthly.to_csv(factor_path + '/accural.csv')
    print('get monthly accural, saved at', factor_path + '/accural.csv')
    return


# 换手率:
#       1. 12month 换手率：过去250日换手率平均值（换手率 = 交易量/发行在外股份数， 发行在外股份数 = 净利润/基本每股收益）
#       2. extra_turnover_20d: 过去20日平均换手率 / 过去250日平均换手率
def get_turnover(factor):
    # select_sql_volume = 'SELECT `date`, `code`, `volume` FROM `market_rt_daily` WHERE `code` <= 10 and code != 0'
    # select_sql_shares = 'SELECT `date`, `code`, `B002000000`/`B003000000` ' \
    #                     'FROM `report_profit_daily` WHERE `code` <= 10 and code != 0'
    select_sql_volume = 'SELECT `date`, `code`, `volume` FROM `market_rt_daily` WHERE `code` != 0'
    select_sql_shares = 'SELECT `date`, `code`, `B002000000`/`B003000000` ' \
                        'FROM `report_profit_daily` WHERE `code` != 0'
    volume = select_data(select_sql_volume).reset_index().set_index(['date', 'code']).iloc[:, 1:]
    shares = select_data(select_sql_shares).reset_index().set_index(['date', 'code']).iloc[:, 1:]
    print('get volume and shares')
    v_s = pd.merge(volume, shares, left_index=True, right_index=True, how='outer')
    v_s['turnover_daily'] = v_s.values[:, 0] / v_s.values[:, 1]
    print('get daily to')
    turnover_12m = rolling_func(250, v_s.reset_index().set_index('date')[['code', 'turnover_daily']],
                                np.mean, 'turnover_daily', 'turnover_12m_mean')
    if factor in ['to_12m', 'all']:
        print('get to_12m daily')
        turnover_12m_monthly = daily2monthly(turnover_12m)
        turnover_12m_monthly.columns = ['turnover_12m_monthly']
        turnover_12m_monthly.to_csv(factor_path + '/to_12m.csv')
        print('get monthly to_12m, saved at', factor_path + '/turnover_12m.csv')
        del turnover_12m_monthly
    if factor in ['extra_to_20d', 'all']:
        turnover_20d = rolling_func(20, v_s.reset_index().set_index('date')[['code', 'turnover_daily']],
                                np.mean, 'turnover_daily', 'turnover_20d_mean')
        turnover_20d['extra_to_daily'] = turnover_20d['turnover_20d_mean'].values\
                                         / turnover_12m['turnover_12m_mean'].values
        extra_to_monthly = daily2monthly(turnover_20d[['code', 'extra_to_daily']])
        extra_to_monthly.columns = ['extra_to_monthly']
        extra_to_monthly.to_csv(factor_path + '/to_extra_20d.csv')
        print('get monthly to_extra_20d, saved at', factor_path + '/to_extra_20d.csv')
    return


if __name__ == '__main__':
    factor_path = config.get('PATH', 'data_pt') + '/factors'
    # get_bs_related('all')
    # get_turnover('all')
    get_accruals_1()
