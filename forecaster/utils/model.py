from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_with_dt(seq_length=10, input_dim=4):
    model = Sequential([
        LSTM(32, input_shape=(seq_length, input_dim), return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(3)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
