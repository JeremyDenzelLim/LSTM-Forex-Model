import numpy as np
from numpy.random import seed

from sklearn.linear_model import LogisticRegression
from sklearn import mixture as mix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import talib as ta


### <summary>
### Basic template algorithm simply initializes the date range and cash. This is a skeleton
### framework you can use for designing an algorithm.
### </summary>
class BasicTemplateAlgorithm(QCAlgorithm):
    '''Basic template algorithm simply initializes the date range and cash'''

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        self.SetStartDate(2018, 10, 15)  # Set Start Date
        self.SetEndDate(2018, 10, 26)  # Set End Date
        self.SetCash(100000)  # Set Strategy Cash
        self.currency = "EURUSD"
        self.symbol = self.AddForex(self.currency, Resolution.Daily).Symbol
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage,
                               AccountType.Margin)  # Set Brokerage Model
        self.SetWarmUp(365)

        self.hist_data = pd.DataFrame(
            self.History([self.currency], 500,
                         Resolution.Daily))  # Asking for historical data
        self.long_list = []
        self.short_list = []
        self.model = SVC(
            C=1.0,
            cache_size=200,
            class_weight=None,
            coef0=0.0,
            decision_function_shape=None,
            degree=3,
            gamma='auto',
            kernel='rbf',
            max_iter=-1,
            probability=True,
            random_state=None,
            shrinking=True,
            tol=0.001,
            verbose=False)
        self.x = 0
        self.atr = 0
        #average ATR over 14 periods
        # self.atr.append(self.ATR(self.symbol, 14))
        # self.Debug("ATR IS: ")
        # self.Debug(self.atr[0])

        self.Schedule.On(
            self.DateRules.EveryDay(self.symbol),
            self.TimeRules.AfterMarketOpen(self.symbol, 1),
            Action(self.Rebalance))

        self.Schedule.On(
            self.DateRules.EveryDay(self.symbol),
            self.TimeRules.Every(TimeSpan.FromMinutes(10)),
            Action(self.Rebalance2))

    def OnData(self, data):
        if self.IsWarmingUp:
            return
        # self.Debug("ATR IS: ")
        # self.Debug(self.atr[0])
        self.hist_data = pd.DataFrame(
            self.History([self.currency], 500,
                         Resolution.Daily))  # Asking for historical data
        df1 = self.hist_data
        self.atr = ta.ATR(df1['high'], df1['low'], df1['close'], timeperiod=14)

    def Rebalance(self):
        new_hist = self.History([
            self.symbol,
        ], 500, Resolution.Minute)
        data_new = pd.DataFrame(new_hist)
        dummy = data_new.tail(1)
        self.hist_data = pd.concat([self.hist_data, dummy])

        if not self.hist_data.empty:
            df = self.hist_data

            df = df[['open', 'high', 'low', 'close']]
            df['High'] = df['high'].shift(1)
            df['Low'] = df['low'].shift(1)
            df['Close'] = df['close'].shift(1)
            df['RSI'] = ta.RSI(np.array(df['Close']), timeperiod=14)
            df['SMA'] = df['Close'].rolling(window=50).mean()
            df['Corr'] = df['SMA'].rolling(window=10).corr(df['Close'])
            df['SAR'] = ta.SAR(
                np.array(df['High']), np.array(df['Low']), 0.2, 0.2)
            # df['ADX'] = ta.ADX(
            #     np.array(df['High']),
            #     np.array(df['Low']),
            #     np.array(df['Close']),
            #     timeperiod=14)
            df.dropna(inplace=True)
            df['Return'] = np.log(df['open'].shift(-1) / df['open'])
            df['Signal'] = 0
            df.loc[df['Return'] > 0, 'Signal'] = 1
            df.loc[df['Return'] < 0, 'Signal'] = 0

            df = df.drop(['high', 'low', 'close'], axis=1)

            ss = MinMaxScaler()

            columns = df.columns.drop(['Return', 'Signal'])
            df[columns] = ss.fit_transform(df[columns])

            self.Debug(df)

            X = df.drop(['Signal', 'Return'], axis=1)
            y = df['Signal']

            self.model.fit(X.iloc[:-1, :], y[:-1])

            output = self.model.predict(X.iloc[-1:, :])
            if output == 1:
                self.Debug("buy")
            else:
                self.Debug("sell")

        if output == 1 and self.currency not in self.long_list and self.currency not in self.short_list:
            self.SetHoldings(self.currency, 1)
            self.long_list.append(self.currency)
            self.Debug("long")

        if output == 0 and self.currency not in self.long_list and self.currency not in self.short_list:
            self.SetHoldings(self.currency, -1)
            self.short_list.append(self.currency)
            self.Debug("short")

    def Rebalance2(self):
        curr_price = self.Securities["EURUSD"].Price
        

        if self.currency in self.long_list:
            cost_basis = self.Portfolio[self.currency].AveragePrice
            # self.Debug("cost basis is " +str(cost_basis))
            if ((curr_price <= float(cost_basis) - float(self.atr[-1]))
                    or (curr_price >=
                        float(self.atr[-1] * 1.25) + float(cost_basis))):
                # self.Debug("SL-TP reached")
                self.Debug("price is" + str(curr_price))
                # If true then sell
                self.SetHoldings(self.currency, 0)
                self.long_list.remove(self.currency)
                self.Debug("squared long")
                # self.Rebalance()

        if self.currency in self.short_list:
            cost_basis = self.Portfolio[self.currency].AveragePrice
            
            if ((curr_price <= float(cost_basis) - float(self.atr[-1] * 1.25))
                    or
                (curr_price >= float(self.atr[-1]) + float(cost_basis))):
                
                self.Debug("price is" + str(curr_price))
                self.SetHoldings(self.currency, 0)
                self.short_list.remove(self.currency)
                self.Debug("squared short")
