import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader.data as web


#Environment params
TIMESTEP = 3
FEATSNUM = 7
TRAIN_TEST = "2019-05-03"
#eur xau sp nasdaq usd
INSTRUMENTNUM = 5
INSTRUMENTS = ("EURUSD=X", "GC=F", "^GSPC", "^IXIC", "USD")


class Market:
    def __init__(self, startValueOfGoods : float = 1000):
        self.currentDate = TIMESTEP 
        #self.nextSellDate = int(self.currentDate + TIMESTEP/8)
        self.totalOfGoods = startValueOfGoods
        self._totalOfGoodsCopy = startValueOfGoods
        self._last = 0
        self.bigData = None
        #firstDayPrices only modified in _retrieveData method
        self.firstDayPrices = list ()
        
        #to set some values
        self._initialize()

        self.history = {"day":[], "basket": [], "reward": [], "totalOfGoods": [], "netProfit": [], "weights": []}

    def _initialize(self) -> None:
        """
        set some values for initialization
        """
        self._startDate = 0

        self.basket = np.zeros(INSTRUMENTNUM)
        self.weights = np.zeros(INSTRUMENTNUM)

        self.weights[INSTRUMENTNUM-1] = 1
        self.basket[INSTRUMENTNUM-1] = self._totalOfGoodsCopy


    def reset(self):
        #to set some values as it starts first time
        self._initialize()

        state = self._getStateForCurrentDateV2()
        return state
    
    def _getData(self) -> None:
        self.bigData = self._retrieveData()
        self._last = len(self.bigData[0])
    
    def _normalize(self, df : pd.DataFrame) -> None:
        #for col in df.columns:
        #    norm = np.linalg.norm(df[col])
        #    df[col] /= norm
        norm = np.linalg.norm(df["Adj Close"])
        df["Adj Close"] /= norm


    def _retrieveData(self, start:dt.datetime = dt.datetime(2000,1,1)) -> list:
        """
        burada çıktı bir dataframeler listesi 
        listenin içindeki her dataframe başlangıc tarihinden itibaren o indexteki şirket için verileri tutuyor
        buradan şirketlere ulaşmak için "_getTicketsOfSP500" fonksiyonu ile indexler üzerinden şirket ismine ulaşılabilir
        """
        
        res = list ()
        self.firstDayPrices = list ()
        ####################
        #debug
        #print("+"*20)
        #print("day: ", self.currentDate - 1)
        ####################
        for instrument in INSTRUMENTS:
            if (instrument == "USD"):
                continue

            df = web.DataReader(instrument, "yahoo", start)
            self.firstDayPrices.append(df.iloc[self.currentDate - 1, :])
            res.append(df)

            ####################
            #debug
            #print("*"*20)
            #print(instrument)
            #print(df.iloc[self.currentDate - 1, :])
            #print("-"*20)
            ####################
            
        return res

    def _getStateForCurrentDate(self) -> np.array:
        """
        returns state for current date which is a list of data frames that holds market info from _startDate to currentDate
        """
        state = list ()
        for instrumentDF in self.bigData:
            df = instrumentDF.iloc[self._startDate:self.currentDate,:].copy()
            #print(df)
            self._prepareDataFrame(df)
            self._normalize(df)
            state.append(df)
        return np.array(state).reshape((-1,TIMESTEP,FEATSNUM,1))


    ###########################################
    ###########################################
    #it is modified to ensure that all prices normalize relativee to first day (actually one day before first buy day)
    ###########################################
    ###########################################
    def _getStateForCurrentDateV2(self) -> np.array:
        """
        returns state for current date which is a list of data frames that holds market info from _startDate to currentDate
        """
        state = list ()
        for instrumentIndex in range(len(self.bigData)):
            instrumentDF = self.bigData[instrumentIndex]
            df = instrumentDF.iloc[self._startDate:self.currentDate,:].copy()
            #print(df)
            self._prepareDataFrameV2(df, instrumentIndex)
            #print("len(df):", len(df))
            #print("len(df.columns):", len(df.columns))
            state.append(df)
        #return np.array(state).reshape((-1,INSTRUMENTNUM-1,TIMESTEP,FEATSNUM))
        return np.array(state).reshape((INSTRUMENTNUM-1,TIMESTEP,FEATSNUM))


    def _incrementAndCheck(self) -> bool:
        """
        This method increment currentDate and _startDate and returns a bool value
        if new currentDate value is either more than or equal to _last day in our data then it returns False
        else it returns True
        """
        self.currentDate += 1
        self._startDate =  int(self.currentDate - TIMESTEP)
        return False if(self.currentDate >= self._last - 1) else True

    def _prepareDataFrame(self, df : pd.DataFrame) -> None:
        df["H/L"] = df["High"] / df["Low"]
        df["H-Step"] = (df["High"] - df["Low"]) / df["Low"]
        df["C/O"] = df["Close"] / df["Open"]
        df["C-Step"] = (df["Close"] - df["Open"] ) / df["Open"]
        df.drop(["Volume", "High", "Close", "Low", "Open"],axis=1 , inplace= True)

    ###########################################
    ###########################################
    #it is modified to ensure that all prices normalize relativee to first day (actually one day before first buy day)
    ###########################################
    ###########################################
    def _prepareDataFrameV2(self, df : pd.DataFrame, index : int) -> None:
        df["H"] = df["High"] / self.firstDayPrices[index]["High"]
        df["L"] = df["Low"] / self.firstDayPrices[index]["Low"]
        df["H-Step"] = (df["High"] - df["Low"]) / df["Low"]
        df["C"] = df["Close"] / self.firstDayPrices[index]["Close"] 
        df["O"] = df["Open"] / self.firstDayPrices[index]["Open"]
        df["C-Step"] = (df["Close"] - df["Open"] ) / df["Open"]
        df["Adj Close"] = df["Adj Close"] / self.firstDayPrices[index]["Adj Close"]
        df.drop(["Volume", "High", "Close", "Low", "Open"],axis=1 , inplace= True)

    def setCurrentDate(self, newCurrentDate) -> None:
        """
        For debugging
        """
        self.currentDate = newCurrentDate
        self._startDate = self.currentDate - (TIMESTEP )

    def _calculateReward(self):
        """
        This method calculates reward for "currentDate"
        """
        reward = list ()
        #for instrument, weight in zip(self.bigData, self.basket):
        #    closePrice = instrument.iloc[self.currentDate, 3]
        #    reward.append(closePrice * weight)
 
        return reward

    def _sellAllInstruments(self) -> float:
        """
        this method sells all instruments and adds profit to "totalOfGoods"
        """
        profit = 0
        #for instrument, weight in zip(self.bigData, self.basket):
        #    closePrice = instrument.iloc[self.currentDate, 3]
        #    profit += closePrice * weight
        
        #because last one is usd
        for index in range(INSTRUMENTNUM-1):
            closePrice = self.bigData[index].iloc[self.currentDate, 3]
            profit += closePrice * self.basket[index]
            self.basket[index] = 0. 
        
        index += 1
        self.totalOfGoods = profit + self.basket[index] if (profit + self.basket[index] != 0) else self.totalOfGoods
        return profit

    def _buyAllInstruments(self) -> float:
        """
        this method buys all instruments and substruct loss from "totalOfGoods"
        """
        loss = 0
        #for instrument, weight in zip(self.bigData, self.basket):
        #    closePrice = instrument.iloc[self.currentDate, 3]
        #    loss += closePrice * weight
        
        for index in range(INSTRUMENTNUM-1):
            closePrice = self.bigData[index].iloc[self.currentDate - 1, 3]
            unitLoss = self.totalOfGoods * self.weights[index]
            loss += unitLoss
            self.basket[index] = closePrice / unitLoss if ( unitLoss != 0 ) else 0
        
        index += 1
        self.basket[index] = self.totalOfGoods * self.weights[index]
        return loss


    def start(self) ->np.array:
        """
        Starts environment and returns current state
        """
        self._getData()
        startState = self._getStateForCurrentDateV2()
        return startState

    def step(self, newWeights) -> (pd.DataFrame, float, bool):
        """
        step method makes necessary changes to end current date and return next state, reward, and is env done info
        and log it
        our action is new weights corresponding instrument
        """
        
        loss = self._buyAllInstruments()

        profit = self._sellAllInstruments()
        self.weights = newWeights
        #reward = self._calculateReward()
        reward = profit - loss
        done = self._incrementAndCheck()
        nextState = self._getStateForCurrentDateV2()

        #log data
        self.history["day"].append(self.currentDate)
        self.history["basket"].append(self.basket)
        self.history["reward"].append(reward)
        self.history["totalOfGoods"].append(self.totalOfGoods)
        self.history["netProfit"].append(profit - loss)
        self.history["weights"].append(newWeights)
        return nextState, reward, done