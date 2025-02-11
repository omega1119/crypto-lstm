### **BTCUSDT Price Prediction with LSTM**

## **Overview**
This project implements a Long Short-Term Memory (LSTM) model with an attention mechanism to predict Bitcoin (BTC) price movements based on historical market data from Binance candlestick (kline) data.

### **Key Features:**
- Loads and preprocesses BTCUSDT market data.
- Scales and sequences data for LSTM training.
- Implements an LSTM model with an attention mechanism.
- Uses early stopping and learning rate reduction for training.
- Evaluates the model and generates BTC price predictions.
- Visualizes actual vs. predicted prices.

## **Setting Up the Environment**
### **Dependencies**
Ensure you have the following Python packages installed:
```bash
pip install numpy tensorflow pandas scikit-learn mplfinance matplotlib
```

### **Steps to Set Up TensorFlow on Apple Silicon**
1. Install Homebrew from [https://brew.sh](https://brew.sh).
2. Install Miniforge3 for macOS arm64:
    ```bash
    chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
    sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
    source ~/miniforge3/bin/activate
    ```
3. Restart the terminal.
4. Create and activate a Conda environment:
    ```bash
    mkdir tensorflow-test && cd tensorflow-test
    conda create --prefix tf_env python=3.8
    conda activate tf_env
    ```
5. Install TensorFlow and dependencies:
    ```bash
    conda install -c apple tensorflow-deps
    python -m pip install tensorflow-macos tensorflow-metal
    ```
6. Install additional packages:
    ```bash
    conda install jupyter pandas numpy matplotlib scikit-learn
    pip install python-binance python-dotenv yfinance torch torchvision torchaudio
    ```
7. Start Jupyter Notebook and verify TensorFlow installation:
    ```bash
    jupyter notebook
    ```
    ```python
    import tensorflow as tf
    print(f"TensorFlow devices: {tf.config.list_physical_devices()}")
    print(f"TensorFlow version: {tf.__version__}")
    ```

## **Project Structure**
```
project_folder/
│── data/                   # Stores market data
│── models/                 # Stores trained models
│── main.py                 # Main script for running the prediction
├── notebook.ipynb          # Jupyter Notebook explaining the strategy
```

## **Running the Full Pipeline**
The `main()` function handles:
1. **Data Preparation:**
   - Loads market data.
   - Normalizes data and creates sequences.
   - Splits into training and test sets.

2. **Model Training:**
   - Trains the LSTM model with attention.
   - Saves the best-performing model.

3. **Evaluation & Prediction:**
   - Evaluates the trained model.
   - Generates future BTC price predictions.
   - Visualizes results.

```python
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

main()
```

### **Key Takeaways:**
- Automates data processing, training, and evaluation.
- Predicts BTC prices for four future intervals.
- Provides a clear visualization of actual vs. predicted prices.

This function is the **entry point** for executing the complete workflow efficiently. 🚀