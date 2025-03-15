import numpy as np
import pandas as pd
import torch
import joblib
from datetime import datetime
from pandas import Timedelta
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from alpaca_trade_api import REST
from finbert import sentiment_from_model
from lstm import LSTMPredictor, calculate_bollinger_bands, calculate_atr, calculate_rsi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Alpaca API config
API_KEY = "ADD_YOUR_OWN_KEY"
API_SECRET = "ADD_YOUR_OWN_SECRET"
BASE_URL = "ADD_YOUR_OWN_URL"

ALPACA_CONFIG = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

# Load LSTM model and scaler
def load_lstm_model():
    model = LSTMPredictor().to(device)
    model.load_state_dict(torch.load("lstm_model.pth", weights_only=True))
    model.eval()
    scaler = joblib.load("lstm_scaler.joblib")
    return model, scaler

# LSTM prediction function
def predict_movement(prices, highs, lows, model, scaler):
    # Calculate indicators
    upper_band, lower_band = calculate_bollinger_bands(prices)
    atr = calculate_atr(prices, highs, lows)
    rsi = calculate_rsi(prices)
    
    # Fill nan values
    upper_band = pd.Series(upper_band).bfill().values
    lower_band = pd.Series(lower_band).bfill().values
    atr = pd.Series(atr).bfill().values
    rsi = pd.Series(rsi).bfill().values
    
    # Need at least 10 data points
    if len(prices) < 10:
        raise ValueError("Not enough data points for prediction")
    
    # Get last window of data
    input_data = np.column_stack((
        prices[-10:], 
        upper_band[-10:], 
        lower_band[-10:], 
        atr[-10:], 
        rsi[-10:]
    ))
    
    # Reshape and normalize using the scaler
    input_data = scaler.transform(input_data)
    input_tensor = torch.tensor(input_data.reshape(1, 10, 5), dtype=torch.float32).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    # Add predicted price change to last price to get the actual price
    predicted_price = prices.iloc[-1] + prediction
    return predicted_price

class AlgoTrader(Strategy):
    def initialize(self, symbol:str='SPY', risk:float=0.5):
        """
        Initialize the trading strategy
        
        Parameters:
        symbol (str): The stock symbol to trade
        risk (float): Portion of cash to risk on each trade (0.0-1.0)
        """
        self.symbol = symbol
        self.sleeptime = "24H"  # Execute once per day
        self.last_trade = None
        self.risk = risk
        
        # Connect to Alpaca API
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        
        # Load LSTM model and scaler
        self.lstm_model, self.scaler = load_lstm_model()
        
        self.log_message(f"AlgoTrader initialized for {symbol} with risk={risk}")
    
    def position_sizing(self):
        """Calculate position size based on risk tolerance"""
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.risk / last_price)
        return cash, last_price, quantity
    
    def get_dates(self, days):
        """Get today's date and a date `days` ago"""
        today = self.get_datetime()
        past = today - Timedelta(days=days)
        return today.strftime('%Y-%m-%d'), past.strftime('%Y-%m-%d')

    def get_sentiment(self):
        today, past = self.get_dates(8)
        news = self.api.get_news(symbol=self.symbol, start=past, end=today)
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = sentiment_from_model(news)
        return probability, sentiment
    
    def get_lstm_prediction(self):
        """Get price prediction from LSTM model"""
        # Get historical price data for indicators
        try:
            historical_data = self.get_historical_prices(self.symbol, 30, "day")
            prices = historical_data.df['close']
            highs = historical_data.df['high']
            lows = historical_data.df['low']
            
            # Make sure we have enough data
            if len(prices) < 20:
                self.log_message(f"Not enough data points for prediction: {len(prices)}")
                return None, None, None
            
            # Make prediction
            predicted_price = predict_movement(
                prices, highs, lows, 
                self.lstm_model, 
                self.scaler
            )
            
            # Calculate predicted change and percentage
            last_price = prices.iloc[-1]
            price_change = predicted_price - last_price
            change_percent = price_change / last_price * 100
            
            return predicted_price, price_change, change_percent
            
        except Exception as e:
            self.log_message(f"Error in LSTM prediction: {e}")
            return None, None, None
        
    def calculate_dynamic_threshold(self):
        historical_data = self.get_historical_prices(self.symbol, 30, "day")
        # Daily percentage changes
        historical_data.df['pct_change'] = historical_data.df['close'].pct_change() * 100
        volatility = historical_data.df['pct_change'].std()
        threshold = volatility * 0.05  # 5% of daily volatility
        return threshold
    
    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()
        
        # Get LSTM prediction
        lstm_result = self.get_lstm_prediction()
        if lstm_result is None:
            return
        
        predicted_price, price_change, change_percent = lstm_result
        threshold = self.calculate_dynamic_threshold()

        if cash > last_price:
            # buy order
            if sentiment == "positive" and probability > 0.999 or change_percent > threshold:
                # Close any existing sell positions
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.2,
                    stop_loss_price=last_price * 0.95
                )
                self.submit_order(order)
                self.last_trade = "buy"
            # sell order
            elif sentiment == "negative" and probability > 0.999 or change_percent < -threshold:
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * 0.8,
                    stop_loss_price=last_price * 1.05
                )
                self.submit_order(order)
                self.last_trade = "sell"

# Run backtest
if __name__ == "__main__":
    broker = Alpaca(ALPACA_CONFIG)
    strategy = AlgoTrader(broker=broker, parameters={"symbol": "SPY", "risk": 0.5,})
    start_date = datetime(2023, 7, 1)
    end_date = datetime(2024, 12, 31)

    strategy.backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        parameters={
            "symbol": "SPY", 
            "risk": 0.5,
        }
    )
