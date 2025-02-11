from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Flatten, AdditiveAttention, Multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import timedelta

# Paths and Constants
SCRIPT_DIR = Path.cwd()
DATA_FOLDER = SCRIPT_DIR / "data"
MODEL_FOLDER = SCRIPT_DIR / "models"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

TIME_STAMP = 1738439134000
KLINE_INTERVAL = "30m"
SYMBOL = "BTCUSDT"
SEQ_LENGTH = 5

PARQUET_PATH = DATA_FOLDER / f"{TIME_STAMP}/{KLINE_INTERVAL}/{SYMBOL}.parquet"
MODEL_PATH = str(MODEL_FOLDER / f"{TIME_STAMP}/{KLINE_INTERVAL}/{SYMBOL}.h5")

def load_market_data(file_path):
    df = pd.read_parquet(file_path)
    df.sort_values("Open Time", inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

def prepare_data(file_path):
    df = load_market_data(file_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return df_scaled, scaler

def build_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def split_data(X, y, train_ratio=0.8):
    train_size = int(len(X) * train_ratio)
    return (X[:train_size], y[:train_size], X[train_size:], y[train_size:])

def build_lstm(X_train, y_train, model_path):
    input_layer = Input(shape=(X_train.shape[1], 1))
    
    lstm_out = LSTM(50, return_sequences=True)(input_layer)
    lstm_out = LSTM(50, return_sequences=True)(lstm_out)
    
    attention = AdditiveAttention(name='attention_weight')([lstm_out, lstm_out])
    attended_output = Multiply()([lstm_out, attention])
    
    flattened = Flatten()(attended_output)
    dense_out = Dense(1)(flattened)
    
    dropout_out = Dropout(0.2)(dense_out)
    output_layer = BatchNormalization()(dropout_out)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
    ]
    
    model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2, callbacks=callbacks)
    return model

def evaluate_model(model, X_test, y_test):
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")

def predict_prices(model, scaler, scaled_data):
    current_batch = scaled_data[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
    predicted_prices = []
    for _ in range(4):
        next_prediction = model.predict(current_batch)
        current_batch = np.append(current_batch[:, 1:, :], next_prediction.reshape(1, 1, 1), axis=1)
        predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])
    return predicted_prices

def plot_predictions(actual_data, predicted_prices):
    actual_data["Open Time"] = pd.to_datetime(actual_data["Open Time"], unit="ms")
    actual_data.set_index("Open Time", inplace=True)
    last_timestamp = actual_data.index[-1]
    prediction_dates = [last_timestamp + timedelta(minutes=30 * i) for i in range(1, 5)]
    
    predictions_df = pd.DataFrame(index=prediction_dates, data=predicted_prices, columns=['Close'])
    combined_data = pd.concat([actual_data, predictions_df])
    
    plt.figure(figsize=(12,6))
    plt.plot(combined_data.index, combined_data['Close'], linestyle='-', marker='o', color='blue', label='Actual Data')
    plt.plot(predictions_df.index, predictions_df['Close'], linestyle='dashed', marker='o', color='red', label='Predicted Data')
    plt.title("BTCUSDT Price: Full Data + Next 4 Predictions")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

def main():
    scaled_data, scaler = prepare_data(PARQUET_PATH)
    X, y = build_sequences(scaled_data, SEQ_LENGTH)
    X_train, y_train, X_test, y_test = split_data(X, y)
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    model = build_lstm(X_train, y_train, MODEL_PATH)
    evaluate_model(model, X_test, y_test)
    predicted_prices = predict_prices(model, scaler, scaled_data)
    actual_data = load_market_data(PARQUET_PATH)
    plot_predictions(actual_data, predicted_prices)

if __name__ == "__main__":
    main()
