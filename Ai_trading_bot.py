from tensorflow.keras.models import Sequential
import json

class PensiveYellowGreenAnt(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2018, 1, 1)  # Set Start Date
        self.SetEndDate(2022,1,1)
        
        model_key = 'bitcoin_price_predictor'
        if self.ObjectStore.ContainsKey(model_key):
            model_str = self.ObjectStore.Read(model_key)
            config = json.loads(model_str)['config']
            self.model = Sequential.from_config(config)
        self.SetBrokerageModel(BrokerageName.Bitfinex, AccountType.Margin)
        self.SetCash(100000)  # Set Strategy Cash
        self.symbol = self.AddCrypto("BTCUSD", Resolution.Daily).Symbol
        self.SetBenchmark(self.symbol)
       

    def OnData(self, data: Slice):
        if self.GetPrediction() == "Up":
            self.SetHoldings(self.symbol,1)
        else:
            self.SetHoldings(self.symbol,-0.5)
            
    
    def GetPrediction(self):
        df = self.History(self.symbol, 40).loc[self.symbol]
        df_change = df[["open","high","low","close","volume"]].pct_change().dropna()
        model_input = []
        for index, row in df_change.tail(30).iterrows():
            model_input.append(np.array(row))
        model_input = np.array([model_input])
        if round(self.model.predict(model_input)[0][0]) == 0:
            return "Down"
        else:
            return "Up"

