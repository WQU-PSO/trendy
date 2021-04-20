import backtrader as bt
import backtrader.feeds as btfeed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfolio as pf
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from pyfolio import *
from pyswarm import pso
from scipy import stats
sns.set_style("darkgrid")

# code adapted from https://www.backtrader.com/docu/
class BreakoutBacktest:
    def __init__(self, settings, csv_file, params_algo):
        self.csv_file    = csv_file
        self.params_algo = params_algo
        self.settings    = settings

    def strat_start(self, funds=10000, com=0.001, tf=bt.TimeFrame.Minutes, compression=60):
        cerebro = bt.Cerebro()
        cerebro.broker.setcommission(commission=com)
        cerebro.broker.setcash(funds)
        #change this to read_csv maybe
        data = FinModel(dataname=self.csv_file, timeframe=tf, compression=compression)

        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
        cerebro.adddata(data)
        cerebro.addstrategy(Breakout,
                            p_window_len = self.params_algo['p_window_len'],
                            p_window_mul = self.params_algo['p_window_mul'],
                            take_prof    = self.params_algo['take_prof'],
                            stop_loss    = self.params_algo['stop_loss'],
                            order_full   = self.settings['order_full'],
                            status_ord   = self.settings['status_ord'],
                            trades       = self.settings['trades'])

        if self.settings['live']:
            print(f'Start funds: {cerebro.broker.getvalue()}')
        strats = cerebro.run()
        strat1 = strats[0]

        if self.settings['live']:
            print(f'End funds: {cerebro.broker.getvalue()}')

        od = strat1.analyzers.getbyname('returns').get_analysis()
        df = pd.DataFrame(od.items(), columns=['date', 'return'])
        df = df.set_index('date')

        self.stability = self.stabilizer(df['return'])

        if self.settings['live']:
            print('live:')
            print('Return: ' + str((cerebro.broker.getvalue() - funds) / funds * 100) + '%')
            print('Stability:' + str(self.stability))
            print('Top-5 Drawdowns:')
            print(pf.show_worst_drawdown_periods(df['return'], top=5))

        # the folowing code was copied and adated from pyfolio documentation
        if self.settings['plot']:
            # Benchmark
            capital_algo = np.cumprod(1.0 + df['return']) * funds
            benchmark_df = pd.read_csv(self.csv_file)
            benchmark_returns = benchmark_df['<CLOSE>'].pct_change()
            capital_benchmark = np.cumprod(1.0 + benchmark_returns) * funds
            df['benchmark_return'] = benchmark_returns

            # Capital Curves
            plt.figure(figsize=(12, 7))
            plt.plot(np.array(capital_algo), color='blue')
            plt.plot(np.array(capital_benchmark), color='red')
            plt.legend(['Algorithm', 'Buy & Hold'])
            plt.title('Capital Curve')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.show()

            # Drawdown
            plt.figure(figsize=(12, 7))
            pf.plot_drawdown_underwater(df['return']).set_xlabel('Time')
            plt.show()

            # Top-5 Drawdowns
            plt.figure(figsize=(12, 7))
            pf.plot_drawdown_periods(df['return'], top=5).set_xlabel('Time')
            plt.show()

            # Simple Returns
            plt.figure(figsize=(12, 7))
            plt.plot(df['return'], 'blue')
            plt.title('Returns')
            plt.xlabel('Time')
            plt.ylabel('Return')
            plt.show()

            # Return Quantiles
            plt.figure(figsize=(12, 7))
            pf.plot_return_quantiles(df['return']).set_xlabel('Timeframe')
            plt.show()

            # Monthly Returns
            plt.figure(figsize=(12, 7))
            pf.plot_monthly_returns_dist(df['return']).set_xlabel('Returns')
            plt.show()


    # (time series) R-squared of linear fit. Negative means unprofitable.
    def stabilizer(self, rtns):
        if len(rtns) < 2:
            return np.nan
        rtns = np.asanyarray(rtns)
        rtns = rtns[~np.isnan(rtns)]
        cum_rtns = np.log1p(rtns).cumsum()
        r_hat = stats.linregress(np.arange(len(cum_rtns)), cum_rtns)[2]
        if cum_rtns[0] < cum_rtns[-1]:
            return r_hat ** 2
        else:
            return -(r_hat ** 2)


class FinModel(btfeed.GenericCSVData):
    params = (
        ('dtformat', ('%Y-%m-%d')),
        ('tmformat', ('%H:%M:%S')),
        ('datetime', 0),
        ('time', 1),
        ('high', 3),
        ('low', 4),
        ('open', 2),
        ('close', 5),
        ('volume', 6)
    )


class BreakoutLine(bt.Indicator):
    lines  = ('p_up', 'p_down', 'pl_value', 'direction',)
    params = (('p_window_len', 12), ('p_window_mul', 30))

    def once(self, start, end):
        hb_len = self.params.p_window_len * self.params.p_window_mul

        # output lines
        p_up      = self.lines.p_up.array
        p_down    = self.lines.p_down.array
        pl_value  = self.lines.pl_value.array
        pl_direct = self.lines.direction.array

        # dataFrame
        df_can = pd.DataFrame()
        for i in range(start, end):
            df_can = df_can.append({'open':  self.data_open[i],
                                    'high':  self.data_high[i],
                                    'low':   self.data_low[i],
                                    'close': self.data_close[i],
                                    'p_up': False,
                                    'p_down': False,
                                    'pl_value': -1.0,
                                    'direction': 0.0},
                                     ignore_index=True)

        # pivot points
        df_can['hrl'] = df_can['high'].rolling(self.params.p_window_len + 1).apply(np.max, raw=True)
        df_can['lrl'] = df_can['low'].rolling(self.params.p_window_len + 1).apply(np.min, raw=True)
        df_can['hrr'] = df_can['high'][::-1].rolling(self.params.p_window_len + 1).apply(np.max, raw=True)
        df_can['lrr'] = df_can['low'][::-1].rolling(self.params.p_window_len + 1).apply(np.min, raw=True)

        df_can['p_up']   = [True if x.low == x.lrl and x.low == x.lrr else False for x in df_can.itertuples()]
        df_can['p_down'] = [True if x.high == x.hrl and x.high == x.hrr else False for x in df_can.itertuples()]

        for i in range(hb_len, df_can.shape[0]):
            df_temp   = df_can[(i - hb_len):i]
            direction = 0.0
            dir_long  = False
            dir_short = False
            piv_line  = -1.0

            if df_can.shape[0] > 0:
                cpd_ind = -1
                pp_down = np.where(df_temp['p_down'][:(-self.params.p_window_len)] == True)[0]
                if len(pp_down) > 0:
                    cpd_ind = pp_down[-1] + 1

                if cpd_ind > -1:
                    hpcd_down  = df_temp['high'].iloc[cpd_ind - 1]
                    prev_highs = np.where((df_temp['high'][:(cpd_ind - 1)] > hpcd_down) & (
                                df_temp['p_down'][:(cpd_ind - 1)] == True))[0]

                    if len(prev_highs) > 0:
                        op_now   = df_temp['open'].iloc[-1]
                        cp_now   = df_temp['close'].iloc[-1]
                        high_pd  = df_temp['high'].iloc[prev_highs[-1]]
                        ind_pg_d = prev_highs[-1] + 1
                        bars_dp  = cpd_ind - ind_pg_d
                        bars_dn  = hb_len - cpd_ind

                        # Calculate trend line and decision
                        slope    = (hpcd_down - high_pd) / bars_dp
                        piv_line = hpcd_down + slope * bars_dn
                        if slope < 0 and cp_now > piv_line > op_now:
                            dir_long  = True
                            direction = 1.0

            # Short position
            if df_can.shape[0] > 0:
                cp_ind_up = -1
                p_piv_up  = np.where(df_temp['p_up'][:(-self.params.p_window_len)] == True)[0]
                if len(p_piv_up) > 0:
                    cp_ind_up = p_piv_up[-1] + 1

                if cp_ind_up > -1:
                    lpc_piv_up = df_temp['low'].iloc[cp_ind_up - 1]
                    prev_lows = np.where((df_temp['low'][:(cp_ind_up - 1)] < lpc_piv_up)
                                & (df_temp['p_up'][:(cp_ind_up - 1)] == True))[0]

                    if len(prev_lows) > 0:
                        op_now  = df_temp['open'].iloc[-1]
                        cp_now  = df_temp['close'].iloc[-1]
                        lppgpu  = df_temp['low'].iloc[prev_lows[-1]]
                        ipb_up  = prev_lows[-1] + 1
                        bars_up = cp_ind_up - ipb_up
                        bar_now = hb_len - cp_ind_up

                        # Calculate trend line & decision
                        slope    = (lpc_piv_up - lppgpu) / bars_up
                        piv_line = lpc_piv_up + slope * bar_now
                        if slope > 0 and cp_now < piv_line < op_now:
                            dir_short = True
                            direction = -1.0

            if not dir_long or not dir_short:
                df_can['pl_value'].iloc[i]  = piv_line
                df_can['direction'].iloc[i] = direction

        # output indicator lines
        for i in range(start, end - 1):
            p_up[i]      = df_can['p_up'][i]
            p_down[i]    = df_can['p_down'][i]
            pl_value[i]  = df_can['pl_value'][i]
            pl_direct[i] = df_can['direction'][i]

# Create Strategy
class Breakout(bt.Strategy):
    params = (
        ('p_window_len', 12),
        ('p_window_mul', 30),
        ('take_prof', 0.08),
        ('stop_loss', 0.15),
        ('order_full', False),
        ('status_ord', False),
        ('trades', False)
    )

    def log(self, text, dt=None):
        dt = self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {text}')

    def __init__(self):
        self.pivot_points = BreakoutLine(self.data,
                                           p_window_len = self.params.p_window_len,
                                           p_window_mul = self.params.p_window_mul)

        self.data_open  = self.datas[0].open
        self.data_high  = self.datas[0].high
        self.data_low   = self.datas[0].low
        self.data_close = self.datas[0].close

    # Trade Event
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        if self.params.trades:
            self.log(f'OP PROFIT, GROSS: {trade.pnl}, NET: {trade.pnlcomm}')

    # Order Event
    def notify_order(self, order):
        if self.params.order_full:
            print('ORDER INFO: \n' + str(order))
        if order.status in [order.Submitted, order.Accepted]:
            return

        # Order completed?
        if order.status in [order.Completed]:
            if order.isbuy():
                if self.params.status_ord:
                    self.log(f'BUY EXECUTED: {str(order.executed.price)}, SIZE: {str(order.executed.size)}')
            if order.issell():
                if self.params.status_ord:
                    self.log(f'SELL EXECUTED: {str(order.executed.price)}, SIZE: {str(order.executed.size)}')

        if order.status in [order.Canceled]:
            if self.params.status_ord:
                self.log('ORDER STATUS: Canceled')
        if order.status in [order.Margin]:
            if self.params.status_ord:
                self.log('ORDER STATUS: Margin')
        if order.status in [order.Rejected]:
            if self.params.status_ord:
                self.log('ORDER STATUS: Rejected')
        if order.status in [order.Partial]:
            if self.params.status_ord:
                self.log('ORDER STATUS: Partial')

    def next(self):
        if self.pivot_points.direction[0] == 1.0 and self.position.size == 0.0:
            self.order_target_percent(target=1.0, exectype=bt.Order.Market)
            if self.params.status_ord:
                self.log(f'BUY CREATE ORDER, {self.data_close[0]}')
        if self.pivot_points.direction[0] == -1.0 and self.position.size == 0.0:
            self.order_target_percent(target=-1.0, exectype=bt.Order.Market)
            if self.params.status_ord:
                self.log(f'SELL CREATE ORDER, {self.data_close[0]}')

        # LONG
        # Checking stop_loss and take_prof
        if self.position.size > 0:
            sl = self.params.take_prof * self.params.stop_loss
            if self.position.price * (1.0 + self.params.take_prof) < self.data_high[0]:
                self.order_target_percent(target=0.0, exectype=bt.Order.Market)
                if self.params.status_ord:
                    self.log(f'CLOSE LONG BY TAKE_PROF CREATE ORDER, {self.data_close[0]}')
            if self.position.price * (1.0 - sl) > self.data_low[0]:
                self.order_target_percent(target=0.0, exectype=bt.Order.Market)
                if self.params.status_ord:
                    self.log(f'CLOSE LONG BY STOP_LOSS CREATE ORDER, {self.data_close[0]}')

        # SHORT
        if self.position.size < 0:
            sl = self.params.take_prof * self.params.stop_loss
            if self.position.price * (1.0 - self.params.take_prof) > self.data_low[0]:
                self.order_target_percent(target=0.0,exectype=bt.Order.Market)
                if self.params.status_ord:
                    self.log(f'CLOSE SHORT BY TAKE_PROF CREATE ORDER, {self.data_close[0]}')
            if self.position.price * (1.0 + sl) < self.data_high[0]:
                self.order_target_percent(target=0.0, exectype=bt.Order.Market)
                if self.params.status_ord:
                    self.log(f'CLOSE SHORT BY STOP_LOSS CREATE ORDER, {self.data_close[0]}')

        # Close if it's reverse direction
        if (self.position.size > 0 and self.pivot_points.direction[0] == -1.0) or (
                self.position.size < 0 and self.pivot_points.direction[0] == 1.0):
            self.order_target_percent(target=0.0, exectype=bt.Order.Market)
            if self.params.status_ord:
                self.log(f'CLOSE POSITION BY REVERSE SIGNAL, {self.data_close[0]}')


def objective_function(x):
    print('+--------------------------------------+')
    print('')
    print('Started with ' + str(x))

    ap = {'p_window_len': int(x[0]),'p_window_mul': int(x[1]),'take_prof': x[2],'stop_loss': x[3]}
    os = {'order_full': False,'status_ord': False,'trades': False,'live': True,'plot': False}

    backtest = BreakoutBacktest(csv_file ='data/aapl_train.csv', params_algo=ap, settings=os)
    backtest.strat_start(funds=10000, com=0.001, tf=bt.TimeFrame.Minutes, compression=60)
    print('')
    return -backtest.stability


# code adapted from https://pythonhosted.org/pyswarm/
lb = [2, 10, 0.01, 0.1]
ub = [120, 100, 0.2, 1.5]
xopt, fopt = pso(objective_function, lb, ub, swarmsize=20, maxiter=40)
print('OPTIMAL PARAMETERS:')
print(xopt, fopt)

params_algo = {'p_window_len': int(xopt[0]),'p_window_mul': int(xopt[1]),
               'take_prof': xopt[2],'stop_loss': xopt[3],}
settings = {'order_full': False,'status_ord': False, 'trades': False, 'live': True,'plot': True}

# Run the strategy with best params

for file in ['/data/aapl_train.csv',
             '/data/aapl_test.csv',
             '/data/aapl_full.csv']:
    print('Launched backtest for ' + file)
    backtest = BreakoutBacktest(csv_file=file,params_algo=params_algo,settings=settings)
    backtest.strat_start(funds=1000, com=0.0004, tf=bt.TimeFrame.Minutes, compression=60)
