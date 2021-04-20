from main import *

params_algo = {'p_window_len': ,
               'p_window_mul': ,
               'take_prof': ,
               'stop_loss': ,
               }

# Just output parameters
settings = {'order_full': False,
                   'status_ord': True,
                   'trades': True,
                   'live': True,
                   'plot': True
                   }

backtest = BreakoutBacktest(csv_file='./data/aapl_full.csv', params_algo=params_algo, settings=settings)
backtest.strat_start(funds=10000, com=0.001, tf=bt.TimeFrame.Minutes, compression=60)
