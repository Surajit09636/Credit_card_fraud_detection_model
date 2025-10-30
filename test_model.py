import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1️⃣  Device Configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# -------------------------------
# 2️⃣  Allow Scaler to Load Safely
# -------------------------------
torch.serialization.add_safe_globals([StandardScaler])

# -------------------------------
# 3️⃣  Load the Saved Model and Scaler
# -------------------------------
checkpoint_path = "./Models/Credit_fraud.pth"
checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

# -------------------------------
# 4️⃣  Define the Same Architecture Used in Training
# -------------------------------
class CreditNN(nn.Module):
    def __init__(self, input_dim):
        super(CreditNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# -------------------------------
# 5️⃣  Rebuild Model and Load Weights
# -------------------------------
scaler = checkpoint["scaler"]
model = CreditNN(input_dim=len(scaler.mean_))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# -------------------------------
# 6️⃣  Create a Sample Transaction
# -------------------------------
# Example transaction (same order as your training dataset, excluding 'Time' and 'Class')
sample = np.array([
    1.19185711131486, 0.26615071205963, 0.16648011335321, 0.448154078460911,
    0.0600176492822243, -0.0823608088155687, -0.0788029833323113, 0.0851016549148105,
    -0.255425128109186, -0.166974414004614, 1.61272666105479, 1.06523531137287,
    0.48909501589608, -0.143772296441519, 0.635558093258208, 0.463917041022171,
    -0.114804663102346, -0.183361270123994, -0.145783041325259, -0.0690831352230203,
    -0.225775248033138, -0.638671952771851, 0.101288021253234, -0.339846475529127,
    0.167170404418143, 0.125894532368176, -0.00898309914322813, 0.0147241691924927,
    20000000.00
])

# -------------------------------
# 7️⃣  Scale and Predict
# -------------------------------
sample_scaled = scaler.transform([sample])
sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    output = model(sample_tensor)
    pred = (output.item() > 0.5)

# -------------------------------
# 8️⃣  Print the Result
# -------------------------------
print("------------------------------------------------")
print(f"🔍 Model Output: {output.item():.4f}")
print(f"✅ Prediction: {'⚠️ FRAUD DETECTED' if pred else '💳 Legitimate Transaction'}")
print("------------------------------------------------")
