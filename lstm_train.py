import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Read the training dataset
train_file_path = './data/train/part-0.parquet'
train_data = pd.read_parquet(train_file_path)
print("Training data preview:")
print(train_data.head())

# Select features and target variable
responders = [col for col in train_data.columns if col.startswith('responder_')]
target = 'responder_6'
# TODO: feature selection
features = [col for col in train_data.columns if col not in responders]

X = train_data[features].copy()
y = train_data[target].copy()

print("Features preview:")
print(X.head())

# Fill NaN values in features with column mean
X.fillna(X.mean(), inplace=True)

# Drop rows where target variable y is NaN
mask = ~y.isna()
X = X[mask]
y = y[mask]

print("Cleaned features preview:")
print(X.head())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences for LSTM
sequence_length = 5

def create_sequences(X, y, seq_len):
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y.iloc[i+seq_len])  # Use .iloc since y might be a Pandas Series
    return torch.tensor(np.array(X_seq), dtype=torch.float32), torch.tensor(np.array(y_seq), dtype=torch.float32)

X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)

# Split training and validation sets
split = -1000  # I simply chose last 1000 samples as validation set
X_train, y_train = X_seq[:split], y_seq[:split]
X_val, y_val = X_seq[split:], y_seq[split:]

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.squeeze()

# Get the number of features (last dim) from the input data
# X_train shape is (batch_size, sequence_length, num_features)
input_size = X_train.shape[2]
model = LSTMModel(input_size)
# Mean Squared Error as the loss function, TODO
criterion = nn.MSELoss()
# learning rate can be adjusted
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
epochs = 10  # TBA
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Test on validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save model
model_file_path = "lstm_model.pth"
torch.save(model.state_dict(), model_file_path)
print(f"Model saved to {model_file_path}")

# Load the model and evaluate
loaded_model = LSTMModel(input_size)
loaded_model.load_state_dict(torch.load(model_file_path))
loaded_model.eval()
print(f"Model loaded from {model_file_path}")

# Calculate RMSE on validation set
y_val_pred = []
with torch.no_grad():
    for X_batch, _ in val_loader:
        preds = loaded_model(X_batch)
        y_val_pred.extend(preds.numpy())

val_rmse = mean_squared_error(y_val.numpy(), np.array(y_val_pred)) ** 0.5
print(f"Validation RMSE: {val_rmse:.4f}")

print("Training complete")
