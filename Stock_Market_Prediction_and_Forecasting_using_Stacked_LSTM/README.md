📈 Stock Market Prediction and Forecasting using Stacked LSTM

This project demonstrates how to use Recurrent Neural Networks (RNNs), specifically Stacked LSTMs (Long Short-Term Memory networks), to predict and forecast stock market prices.
We use historical Apple (AAPL) stock price data, apply preprocessing, train a deep learning model, and forecast future trends.

🚀 Project Workflow

Data Collection

Used Apple stock dataset (AAPL.csv).

Focused on the Close price for prediction.

Data Preprocessing

Normalized prices using MinMaxScaler.

Created train-test split (65%-35%).

Converted time-series data into supervised learning format with a look-back window of 100 days.

Model Architecture

Stacked LSTM with three layers:

LSTM(50, return_sequences=True)

LSTM(50, return_sequences=True)

LSTM(50)

Dense(1) for regression output

Dropout layers added to prevent overfitting.

Optimizer: Adam

Loss function: Mean Squared Error (MSE)

Training

Trained for up to 100 epochs.

Early Stopping applied (patience=10).

Batch size: 64.

Evaluation Metrics

Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

R² Score

Results

Plotted train vs test predictions against actual stock prices.

Forecasted next 30 days of stock prices based on last 100 days.

📊 Results

Training RMSE: ~Low error (depends on dataset run).

Testing RMSE: Indicates how well the model generalizes.

The model captures stock price trends but may not perfectly predict exact values (as markets are non-deterministic).

Visualization Outputs:

Stock price history (Apple closing price).

Actual vs Predicted prices (Train & Test).

Next 30 days forecast vs last 100 days.

🛠️ Tech Stack

Python 3

TensorFlow / Keras for deep learning

scikit-learn for preprocessing & metrics

Matplotlib for visualization

Pandas / NumPy for data handling

⚠️ Disclaimer

This project is for educational purposes only.
Stock market prices are influenced by many unpredictable factors, and this model should not be used for real financial trading decisions.

✨ Future Improvements

Experiment with different look-back windows (30, 60, 120 days).

Try GRU networks or hybrid models (CNN+LSTM).

Hyperparameter tuning for deeper optimization.

Incorporate additional features (Open, High, Low, Volume).

🙌 Acknowledgements

Inspired by stock forecasting tutorials and applied deep learning techniques.

Built with TensorFlow, Keras, and scikit-learn.