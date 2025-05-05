import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from forecaster.config import SEQ_LENGTH, TEST_SIZE, DT, EPOCHS, BATCH_SIZE, MODEL_PATH, LOSS_PLOT_PATH
from forecaster.utils.lorenz import generate_lorenz_data, save_data_to_csv
from forecaster.utils.preprocessing import preprocess_data
from forecaster.utils.model import build_lstm_with_dt
from forecaster.utils.visualization import plot_loss_curve, visualize_model

# Step 1: Generate Data
# if not os.path.exists("forecaster/data/lorenz_data.csv"):
_, data, dt = generate_lorenz_data(t_span=(0, 20), t_eval_step=0.001, init_state=[10, 10, 10])
    # save_data_to_csv(data)
# else:
#     dt = DT

# Step 2: Preprocess
scaler = preprocess_data(seq_length=SEQ_LENGTH, test_size=TEST_SIZE, dt=dt)

# Step 3: Load Data
data = np.load("forecaster/data/processed_data_dt.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Step 4: Build and Train Model
model = build_lstm_with_dt(seq_length=X_train.shape[1], input_dim=X_train.shape[2])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Step 5: Save and Plot
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)

os.makedirs(os.path.dirname(LOSS_PLOT_PATH), exist_ok=True)
plot_loss_curve(history, LOSS_PLOT_PATH, "LSTM with dt")

# Step 6: Visualize
visualize_model(MODEL_PATH)