import torch

def infer(model, X_test):
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32)
        predictions = model(X_test)
        return predictions.numpy()
