import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader.data as web

import matplotlib.pyplot as plt

import tensorflow as tf


#Environment params
TIMESTEP = 30
FEATSNUM = 6
TRAIN_TEST = "2019-05-03"
#eur xau sp nasdaq usd
INSTRUMENTNUM = 5
INSTRUMENTS = ("EURUSD=X", "GC=F", "^GSPC", "^IXIC", "USD")

#model
ACTIVATION_ACTOR = "sigmoid"
ACTIVATION_CRITIC = "sigmoid"


class Market:
    def __init__(self, totalOfGoods : float = 1000., howToReorganize = False):
        #hold copy of totalOfGoods for reseting environment
        self._startGoodsValue = totalOfGoods
        self.bigData = list()
        self._initializationOfVariables()
        self._lastDate = 0
        self.howToReorganize = howToReorganize

        

    def _retrieveData(self, start:dt.datetime = dt.datetime(2010,1,1)) -> None:
        """
        Retrieve instuments data from market

        Returns None
        """
        for instrument in INSTRUMENTS:
            if (instrument == "USD"):
                continue
            df = web.DataReader(instrument, "yahoo", start)
            self.bigData.append(df)

        if (self.howToReorganize):
            self.reorganizeBigDataGetAll()
        else:
            self.reorganizeBigData()
    
    def reorganizeBigData(self):
        """
        burada hepsini euro dolara göre aldım
        """

        self.bigData[0] = self.bigData[0].shift(1, freq = "D")
        usaIndex = set(self.bigData[0].index)
        goldIndex = set(self.bigData[1].index)
        londonIndex = set(self.bigData[1].index)

        reIndex = usaIndex.union(goldIndex)
        self.bigData[1] = self.bigData[1].reindex(list(reIndex)).ffill().bfill().sort_index().reindex(self.bigData[0].index)

        reIndex = usaIndex.union(londonIndex)
        self.bigData[2] = self.bigData[2].reindex(list(reIndex)).ffill().bfill().sort_index().reindex(self.bigData[0].index)

        self.bigData[3] = self.bigData[3].reindex(list(reIndex)).ffill().bfill().sort_index().reindex(self.bigData[0].index)

    def reorganizeBigDataGetAll(self):
        """
        nu metodla hepsinin günlerini esitledim
        """
        
        reIndex = set ()
        for df in self.bigData:
            temp = set (df.index)
            reIndex = reIndex.union(temp)
        
        for i in range(len(self.bigData)):
            self.bigData[i] = self.bigData[i].reindex(reIndex).ffill()
            self.bigData[i].bfill(inplace = True)
            self.bigData[i].sort_index(inplace=True)

    def _initializationOfVariables(self) -> None:
        """
        Every time user call reset method _initializationOfVariables method invoked automatically to reset necessary variables
        to their initial values

        Returns None
        """
        
        self.totalOfGoods = self._startGoodsValue
        self.analizeStartDate = 0 
        self.timeInterval = TIMESTEP
        self.currentDate = TIMESTEP

        # basket is a list that holds how much of instruments is in our basket by its unit 
        self.basket = np.zeros(INSTRUMENTNUM)
        self.weights = np.zeros(INSTRUMENTNUM)

        self.weights[INSTRUMENTNUM-1] = 1
        self.basket[INSTRUMENTNUM-1] = self.totalOfGoods
        
    
    def start(self) -> np.array:
        """
        Start environment by retreiving necessary market data.

        Returns state
        """
        self._retrieveData()
        #set last date
        self._lastDate = len(self.bigData[0])
        
        state = self._getCurrentState()
        return state


    def reset(self):
        self._initializationOfVariables()
        state =  self._getCurrentState()
        return state



    def _getCurrentState(self) -> np.array:
        """
        Calculate current state respective to self.currentDate

        Returns current State of market as a np.array
        """
        state = list ()
        for instrumentDF in self.bigData:
            df = instrumentDF.iloc[self.analizeStartDate:self.currentDate,:].copy()
            
            #normalize data and rearrange headers
            self.preprocessRawData(df)
            state.append(df)
        
        #its Shape is (INSTRUMENTNUM - 1,TIMESTEP,FEATSNUM,1) 
        return np.array(state).reshape((-1,TIMESTEP,FEATSNUM,1))

    def preprocessRawData(self, df) -> None:
        # High       Low      Open     Close  Volume  Adj Close
        df["H"] = df.loc[:, "High"] / df.iloc[0, 0]
        df["L"] = df.loc[:, "Low"] / df.iloc[0, 1]
        df["H-Step"] = (df.loc[:, "High"] - df.loc[:, "Low"]) / df.loc[:, "Low"]
        #df["C"] = df.loc[:, "Close"] / df.iloc[0, "Close"] 
        df["O"] = df.loc[:, "Open"] / df.iloc[0, 2]
        df["C-Step"] = (df.loc[:, "Close"] - df.loc[:, "Open"] ) / df.loc[:, "Open"]
        df.loc[:, "Adj Close"] = df.loc[:, "Adj Close"] / df.iloc[0, 5]
        df.drop(["Volume", "High", "Close", "Low", "Open"],axis=1 , inplace= True)


    def _buyAllInstruments(self):
        
        loss : float = 0

        for index in range(INSTRUMENTNUM-1):
            closePrice = self.bigData[index].iloc[self.currentDate-1 , 3]
            unitLoss = self.totalOfGoods * self.weights[index]
            loss += unitLoss
            self.basket[index] =  unitLoss / closePrice if ( unitLoss != 0 ) else 0
        
        #for dolar
        index += 1
        self.basket[index] = self.totalOfGoods * self.weights[index]

        return loss

    def _sellAllInstruments(self) -> float:
        """
        this method sells all instruments and adds profit to "totalOfGoods"
        """
        profit = 0
        #because last one is usd
        for index in range(INSTRUMENTNUM-1):
            closePrice = self.bigData[index].iloc[self.currentDate, 3]
            unitProfit = closePrice * self.basket[index]
            profit += unitProfit
            self.basket[index] = 0. 
        
        index += 1
        self.totalOfGoods = profit + self.basket[index] if (profit + self.basket[index] != 0) else self.totalOfGoods
        return profit

    
    def incrementDay(self):
        """
        This method increment currentDate and _startDate and returns a bool value
        if new currentDate value is either more than or equal to _last day in our data then it returns False
        else it returns True
        """
        self.currentDate += 1
        self.analizeStartDate =  int(self.currentDate - self.timeInterval)

        done = self.currentDate >= self._lastDate - 1
        return done


    def step(self, weights : np.array, normalizedReward : float = False):
        """
        Buy instruments at last days close
        sell them all at close
        increment day

        weights: is an np.array that holds how to disturbue total money to insturuments which is also action
        normalizedReward: if it is true than returns normalized profit else normal profit default False 

        Returns (nextState, reward, done)
        """
        sumOfWeights = tf.math.reduce_sum(weights)
        #if(0.9< sumOfWeights <1.1):
        #    self.weights = np.array(weights).reshape((5,))
        #else:
        #    print(f"Weights has problems their sum is not one\nsum: {sumOfWeights}")
        #    print(f"State {self._getCurrentState()}")
        #    return False
        self.weights = np.array(weights)
        oldTotalOfGoods = self.totalOfGoods
        loss = self._buyAllInstruments()
        self.basketCopy = self.basket.copy()
        profit = self._sellAllInstruments()

        reward = (profit - loss) / oldTotalOfGoods if normalizedReward else profit - loss

        done = self.incrementDay()

        nextState = self._getCurrentState()

        return (nextState, reward, done)






if __name__ == "__main__":
    env = Market()
    startState = env.start()
    print(startState.shape)
    print(startState)
