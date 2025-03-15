import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bollinger Bands
def calculate_bollinger_bands(prices, window=20, num_std_dev=2):
    rolling_mean = pd.Series(prices).rolling(window).mean()
    rolling_std = pd.Series(prices).rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

# Average True Range
def calculate_atr(prices, highs, lows, window=14):
    prices_series = pd.Series(prices)
    tr = pd.DataFrame({'high-low': highs - lows, 
                       'high-close': abs(highs - prices_series.shift(1)),
                       'low-close': abs(lows - prices_series.shift(1))})
    tr = tr.max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

# Relative Strength Index
def calculate_rsi(prices, window=14):
    delta = pd.Series(prices).diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Prepare Data
def prepare_data(prices, highs, lows, window=10): # tried 10, 20 (b/w 10-30 works best for financial data)
    # Calculate indicators
    upper_band, lower_band = calculate_bollinger_bands(prices)
    atr = calculate_atr(prices, highs, lows)
    rsi = calculate_rsi(prices)
    
    # Fill nan values
    upper_band = pd.Series(upper_band).bfill().values
    lower_band = pd.Series(lower_band).bfill().values
    atr = pd.Series(atr).bfill().values
    rsi = pd.Series(rsi).bfill().values

    X = []
    Y = []
    
    for i in range(window, len(prices)):
        price_window = prices[i-window:i]
        upper_band_window = upper_band[i-window:i]
        lower_band_window = lower_band[i-window:i]
        atr_window = atr[i-window:i]
        rsi_window = rsi[i-window:i]
        
        # Stack features
        X.append(np.column_stack((price_window, upper_band_window, lower_band_window, atr_window, rsi_window)))
        # Price change
        Y.append(prices[i] - prices[i-1])

    return np.array(X), np.array(Y)

# LSTM Model
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Last output

# Train the model
def train_lstm(prices, highs, lows):
    # Prepare training data
    X, Y = prepare_data(prices, highs, lows, window=10)

    # Normalize input features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    # Save scaler
    joblib.dump(scaler, "lstm_scaler.joblib")
    # Convert to tensors
    X_train, Y_train = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LSTMPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(50):  # 50 Epochs
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X.to(device)).squeeze()
            loss = loss_fn(outputs, batch_Y.to(device))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "lstm_model.pth")
    print("Training complete! Model saved as lstm_model.pth")

# Download data, train the model, and make a prediction
def main():
    # Stock symbol to train the model on
    symbol = 'SPY' 
    
    # Download historical data flatten
    data = yf.download(symbol, start="2018-01-01", end="2023-12-31")
    prices = data['Close'].values.flatten()
    highs = data['High'].values.flatten()
    lows = data['Low'].values.flatten()
    
    # Train the model
    train_lstm(prices, highs, lows)

if __name__ == "__main__":
    main()
