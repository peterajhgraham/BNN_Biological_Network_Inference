import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

class BayesianNN(nn.Module):
    def __init__(self, input_dim):
        super(BayesianNN, self).__init__()
        self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=input_dim, out_features=64)
        self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=64, out_features=32)
        self.fc3 = bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=32, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def train_model(X_train, y_train, input_dim, epochs=10, learning_rate=0.001):
    model = BayesianNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model
