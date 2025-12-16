import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")


torch.serialization.add_safe_globals([StandardScaler])


checkpoint_path = "./Models/Credit_fraud.pth"
checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)


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


sample = np.array([
    -2.31222654232683, 1.95199201064158, -1.60985073229769, 3.9979055875468,
    -0.522187864667764, -1.42654531920595, -2.53738730624579, 1.39165724829804,
    -2.77008927719433, -2.77227214465915, 3.20203320709635, -2.89990738849473,
    -0.595221881324605, -4.28925378244217, 0.389724120274487, -1.14074717980657,
    -2.83005567450437, -0.0168224681808257, 0.416955705037907, 0.126910559061474,
    0.517232370861764, -0.0350493686052974, -0.465211076182388, 0.320198198514526,
    0.0445191674731724, 0.177839798284401, 0.261145002567677, -0.143275874698919,
    0.0
])


sample_scaled = scaler.transform([sample])
sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    output = model(sample_tensor)
    pred = (output.item() > 0.5)


print("------------------------------------------------")
print(f"🔍 Model Output: {output.item():.4f}")
print(f"✅ Prediction: {'⚠️ FRAUD DETECTED' if pred else '💳 Legitimate Transaction'}")
print("------------------------------------------------")
