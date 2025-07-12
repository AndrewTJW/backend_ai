import yfinance as yf
import pandas as pd
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


def downloadData(stock):
    current_date = date.today()
    data = yf.download(stock, start="2023-01-01", end=current_date, auto_adjust=False)
    if data.empty:
        print("No data found for this symbol.")
        return None
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data

def initiateObv(stock_data):
    close = stock_data['Close'].values
    volume = stock_data['Volume'].values

    obv = [0]  # Start OBV at 0

    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv.append(obv[-1] + volume[i])
        elif close[i] < close[i - 1]:
            obv.append(obv[-1] - volume[i])
        else:
            obv.append(obv[-1])  # No change

    return pd.Series(obv, index=stock_data.index)

def initiateRelativeStrengthIndex(stock_data):
    #graph from 0 - 100
    #if RSI > 70 == overbought means OVERVALUED more risk
    #if RSI < 30 == oversold means UNDERVALUED better
    period = 14 #14 days is standard
    array_of_closing_price = stock_data['Close'].values
    gains = []
    losses = []
    rsi_arr = [None] * period
    for i in range(period):
        change = array_of_closing_price[i + 1] - array_of_closing_price[i]
        if change > 0: #if earn
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change)) #can only take positive magnitude

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:
        RSI = 100
    else:
        avg_gain_divide_avg_loss = average_gain / average_loss
        rsi = 100 - (100 / (1 + avg_gain_divide_avg_loss))
    rsi_arr.append(rsi) #first rsi value

    for i in range(period + 1, len(array_of_closing_price) - 1):
        change = array_of_closing_price[i + 1] - array_of_closing_price[i]
        if change > 0: #if earn
            curr_gain = change
            curr_loss = 0
        else:
            curr_loss = abs(change)
            curr_gain = 0 #can only take positive magnitude

        average_gain = (average_gain * (period - 1) + curr_gain) / period
        average_loss = (average_loss * (period - 1) + curr_loss) / period

        if average_loss == 0:
            rsi = 100
        else:
            average_gain_divide_average_loss = average_gain / average_loss
            rsi = 100 - (100 / (1 + average_gain_divide_average_loss))
        rsi_arr.append(rsi)

    while (len(rsi_arr) < len(array_of_closing_price)):
        rsi_arr.append(np.nan) #fill up empty space
    return pd.Series(rsi_arr, index=stock_data.index)


def FeatureEngineering(stock_data):
    stock_data['Return_1d'] = stock_data['Close'].pct_change()
    stock_data['Volatility'] = stock_data['Return_1d'].rolling(window=5).std()
    stock_data['OBV'] = initiateObv(stock_data)
    stock_data['RSI'] = initiateRelativeStrengthIndex(stock_data)
    stock_data['Target'] = stock_data['Close'].pct_change().shift(-1)
    stock_data = stock_data.dropna()
    print(stock_data)
    initiateRelativeStrengthIndex(stock_data)
    return stock_data


def predict(clean_stock_data):
    X = clean_stock_data[['OBV', 'RSI', 'Volatility', 'Return_1d']]
    Y = clean_stock_data['Target']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)
    model = RandomForestRegressor(n_estimators=1000, random_state=42)
    model.fit(X_train, Y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(Y_test, predictions)
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))

    print("📊 MAE:", round(mae, 4))
    print("📉 RMSE:", round(rmse, 4))

    latest_features = X.iloc[[-1]]
    predicted_return = model.predict(latest_features)[0]

    today_price = clean_stock_data['Close'].iloc[-1].values[0]
    predicted_price = today_price * (1 + predicted_return)

    print(f"📈 Today’s price: ${today_price:.2f}")
    print(f"🔮 Predicted return: {predicted_return * 100:.2f}%")
    print(f"💵 Predicted price for tomorrow: ${predicted_price:.2f}")
    print("🔼 Movement:", "UP" if predicted_return > 0 else "DOWN")


    # Plot actual vs predicted returns
    plt.figure(figsize=(12, 6))
    plt.plot(Y_test.values, label='Actual Return', marker='o')
    plt.plot(predictions, label='Predicted Return', marker='x')
    plt.title('Actual vs Predicted Stock Returns')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return { # returning an object
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'predicted_return_percent': round(predicted_return * 100, 2),
        'predicted_price': round(predicted_price, 2),
        'today_price': round(today_price, 2),
        'direction': 'UP' if predicted_return > 0 else 'DOWN',
        'actual_returns': Y_test.tolist(),
        'predicted_returns': predictions.tolist()
    }




