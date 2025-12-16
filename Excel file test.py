import torch
import torch.nn as nn
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.serialization.add_safe_globals([StandardScaler])

checkpoint = torch.load("./Models/Credit_fraud.pth", weights_only=False, map_location=device)

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

scaler = checkpoint["scaler"]
model = CreditNN(input_dim=len(scaler.mean_))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

input_path = "./Data/input_transactions.xlsx"
output_dir = "./Output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "predicted_transactions.xlsx")

df = pd.read_excel(input_path)
X_scaled = scaler.transform(df.values)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    outputs = model(X_tensor).cpu().numpy()

df["Class"] = (outputs > 0.5).astype(int)
df["Prediction_Label"] = df["Class"].map({0: "Legitimate", 1: "Fraud"})

df.to_excel(output_path, index=False)

print(f"Output saved to {output_path}")
