import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import joblib

def plot_loss_curve(history, filename, model_name):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} Model Loss Curve')
    plt.savefig(filename)
    plt.show()

def visualize_model(model_path):
    model = load_model(model_path)
    scaler = joblib.load("data/scaler.pkl")
    data = np.load("data/processed_data_dt.npz")
    X_test, y_test = data["X_test"], data["y_test"]
    y_pred = model.predict(X_test)

    components = ['dx', 'dy', 'dz']
    plt.figure(figsize=(15, 10))

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(y_test[:, i], label=f'True {components[i]}', alpha=0.7)
        plt.plot(y_pred[:, i], label=f'Predicted {components[i]}', alpha=0.7)
        plt.legend()
        plt.title(f"LSTM Prediction vs Ground Truth ({components[i]})")

    plt.tight_layout()
    plt.show()
