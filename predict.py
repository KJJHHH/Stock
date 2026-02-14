
import numpy as np
import torch

RETURN_SCALE = 100.0

def predict(
    model,
    X_test,
    scaler,
    y_test,
    device=None,
    return_mean=0.0,
    return_std=1.0,
    task="regression",
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_test_device = X_test.to(device)
        predicted = model(X_test_device)

    predicted = predicted.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()
    if task == "classification":
        predicted = 1.0 / (1.0 + np.exp(-predicted))
        print("Predicted vs Actual (probability)")
        for i in range(10):
            idx = -1 - i
            print(f"Predicted: {predicted[idx][0]:.4f}, Actual: {y_test[idx][0]:.0f}")
    else:
        predicted = (predicted * return_std + return_mean) / RETURN_SCALE
        y_test = (y_test * return_std + return_mean) / RETURN_SCALE
        print("Predicted vs Actual (returns)")
        for i in range(10):
            idx = -1 - i
            print(f"Predicted: {predicted[idx][0]:.4f}, Actual: {y_test[idx][0]:.4f}")

    return predicted, y_test

__all__ = ["predict", "predict_sklearn"]


def predict_sklearn(
    model,
    X_test,
    y_test,
    return_mean=0.0,
    return_std=1.0,
    task="regression",
):
    X_np = X_test.reshape(len(X_test), -1)
    y_np = y_test.reshape(-1, 1)
    if task == "classification":
        if hasattr(model, "predict_proba"):
            predicted = model.predict_proba(X_np)[:, 1].reshape(-1, 1)
        else:
            logits = model.decision_function(X_np)
            predicted = 1.0 / (1.0 + np.exp(-logits)).reshape(-1, 1)
    else:
        predicted = model.predict(X_np).reshape(-1, 1)
        predicted = (predicted * return_std + return_mean) / RETURN_SCALE
        y_np = (y_np * return_std + return_mean) / RETURN_SCALE
    return predicted, y_np


