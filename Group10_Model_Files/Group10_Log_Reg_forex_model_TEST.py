import numpy as np
import numpy as np
from numpy.random import seed

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd


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
            self.History([self.currency], 365,
                         Resolution.Daily))
        self.long_list =[]
        self.short_list =[]
        #self.model = LogisticRegression()
        self.x = 0

        self.Schedule.On(self.DateRules.EveryDay(self.symbol),self.TimeRules.AfterMarketOpen(self.symbol,10),Action(self.Rebalance)) 
        
        self.Schedule.On(self.DateRules.EveryDay(self.symbol),self.TimeRules.Every(TimeSpan.FromMinutes(10)),Action(self.Rebalance2))
        
    def OnData(self, data):
        if self.IsWarmingUp:
            return
        
        if self.x==0:  #To make sure the model is build only once and avoid computation at every new data
            currency_data  = self.hist_data
            L= len(currency_data) # Checking the length of data
        
            if not currency_data.empty: 
                data = pd.DataFrame(currency_data.close)  #Get the close prices. Also storing as dataframe
                #Data Preparation for input to Logistic Regression
                stored = {} # To prepare and store data
                for i in range(11): # For getting 10 lags ...Can be increased if morr lags are required
                    stored['EURUSD_lag_{}'.format(i)] = data.shift(i).values[:,0].tolist() #creating lags

                stored = pd.DataFrame(stored)
                stored = stored.dropna() # drop na values
                stored = stored.reset_index(drop=True)
        
                stored["Y"] = stored["EURUSD_lag_0"].pct_change()# get the percent change from previous time
        
                for i in range(len(stored)): # loop to make Y as categorical
                    if stored.loc[i,"Y"] > 0:
                        stored.loc[i,"Y"] = "UP"
                    else:
                        stored.loc[i,"Y"] = "DOWN"
               
                X_data = stored.iloc[:,np.r_[1:11]]  # extract only lag1, Lag2, lag3.. As Lag 0 is the data itself and will  not be available during prediction
                #self.Debug( "X data is" +str(X_data))
            
                Y_data = stored["Y"]
                
                               #USE TimeSeriesSplit to split data into n sequential splits
                tscv = TimeSeriesSplit(n_splits=2)
                
                # Make cells and epochs to be used in grid search.
                cells = [100,200]
                epochs  = [100,200]
                
                # creating a datframe to store final results of cross validation for different combination of cells and epochs
                df = pd.DataFrame(columns= ['cells','epoch','mse'])
                
                #Loop for every combination of cells and epochs. In this setup, 4 combinations of cells and epochs [100, 100] [ 100,200] [200,100] [200,200]
                for i in cells:
                    for j in epochs:
                        
                        cvscores = []
                        # to store CV results
                        #Run the LSTM in loop for every combination of cells an epochs and every train/test split in order to get average mse for each combination.
                        for train_index, test_index in tscv.split(X_data):
                            #self.Debug("TRAIN:", train_index, "TEST:", test_index)
                            X_train, X_test = X_data[train_index], X_data[test_index]
                            Y_train, Y_test = Y_data[train_index], Y_data[test_index]
                            
                            X_train= np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
                            
                            X_test= np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))
                            
                # self.Debug( "Y data is" +str(Y_data))
                 # Number of trees in random forest
                solver = ['liblinear','saga','newton-cg','sag']
                # Number of features to consider at every split
                #multi_class = ['auto']
                # Maximum number of levels in tree
                #penalty = ['l1' , 'l2']
                # Minimum number of samples required to split a node
                max_iter = [50,75,100,125]

                # Create the random grid
                rf = LogisticRegression(solver = 'liblinear')
                random_grid = {
                               'solver' : solver,
                               'max_iter' : max_iter}
                rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 5, verbose=2, random_state=42, n_jobs = -1)
                self.model = rf_random
                
                self.model.fit(X_data,Y_data)
                score = self.model.score(X_data, Y_data)
                # self.Debug("Train Accuracy of final model: " + str(score))
                
                # To get the coefficients from model
                #A = pd.DataFrame(X_data.columns)
                #B = pd.DataFrame(np.transpose(self.model.coef_))
                #C =pd.concat([A,B], axis = 1)
                # self.Debug("The coefficients are: "+ str(C))
                
                self.x=1     # End the model
                
        # self.Rebalance()
                

    def Rebalance(self):
        old_hist = self.History([self.currency], 1000, Resolution.Daily) # Asking for historical data
        new_hist = self.History([self.currency], 1000, Resolution.Minute) # Asking for historical data
        
        
        L= len(new_hist) # Checking the length of data
        
        data_old = pd.DataFrame(old_hist.close)
        data_new = pd.DataFrame(new_hist.close)  #Get the close prices. Also storing as dataframe
        dummy = data_new.tail(1)
        data = pd.concat([data_old,dummy])
        self.Debug(data)
        
        #Prepare test data similar way as earlier
        test = {}
        for i in range(10):
            test['EURUSD_lag_{}'.format(i+1)] = data.shift(i).values[:,0].tolist()

        test = pd.DataFrame(test)
        test = pd.DataFrame(test.iloc[-1, :]) # take the last values 
        test = pd.DataFrame(np.transpose(test)) # transose to get in desired model shape        
        
        output = self.model.predict(test)
        self.Debug("Output from LR model is" + str(output))
        
        if output == "UP"  and self.currency not in self.long_list and self.currency not in self.short_list :
            #self.Debug("output is greater")
            # Buy the currency with X% of holding in this case 90%
            self.SetHoldings(self.currency, 1)
            self.long_list.append(self.currency)
            self.Debug("long")
        
        if output == "DOWN"  and self.currency not in self.long_list and self.currency not in self.short_list:
            #self.Debug("output is lesser")
            # Buy the currency with X% of holding in this case 90%
            self.SetHoldings(self.currency, -1)
            self.short_list.append(self.currency)
            self.Debug("short")        
        
    def Rebalance2(self):
        
        curr_data = self.History([self.currency], 120, Resolution.Minute) # Asking for historical data
        
        curr_price = curr_data.close[-1]
        self.Debug(curr_price)
        
        if self.currency in self.long_list:
            cost_basis = self.Portfolio[self.currency].AveragePrice
            self.Debug("cost basis is " +str(cost_basis))
            if  ((curr_price <= float(0.99) * float(cost_basis)) or (curr_price >= float(1.01) * float(cost_basis))):
                #self.Debug("SL-TP reached")
                #self.Debug("price is" + str(price))
                #If true then sell
                self.SetHoldings(self.currency, 0)
                self.long_list.remove(self.currency)
                self.Debug("squared long")
                self.Rebalance()
        
        if self.currency in self.short_list:
            cost_basis = self.Portfolio[self.currency].AveragePrice
            self.Debug("cost basis is " +str(cost_basis))
            if  ((curr_price <= float(0.99) * float(cost_basis)) or (curr_price >= float(1.01) * float(cost_basis))):
                #self.Debug("SL-TP reached")
                #self.Debug("price is" + str(price))
                #If true then buy back
                self.SetHoldings(self.currency, 0)
                self.short_list.remove(self.currency)
                self.Debug("squared short")
                #self.Debug("END: Ondata")
                self.Rebalance()
