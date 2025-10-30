# 💳 Credit Card Fraud Detection Model

## 📘 Overview
This project uses a **Neural Network** to detect fraudulent transactions in credit card data.  
The dataset contains anonymized transaction features (`V1`–`V28`), `Amount`, and a `Class` label where:
- `0` → Legitimate transaction
- `1` → Fraudulent transaction

---

## 🧠 Model Workflow

### 1️⃣ Data Preprocessing
- Drop unnecessary columns: `Time` (not relevant for fraud detection).
- Split data:
  ```python
  X = df.drop(['Class', 'Time'], axis=1)
  y = df['Class']
  ```
- Scale input features using `StandardScaler`.
- Handle data imbalance using `RandomOverSampler` from `imblearn`.

### 2️⃣ Model Architecture (FraudNet)
A simple **Fully Connected Neural Network (Feedforward NN)**:
```python
class FraudNet(nn.Module):
    def __init__(self, input_size):
        super(FraudNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)
```

### 3️⃣ Training
- Optimizer: `Adam`
- Loss: `BCELoss`
- Batch Size: 64
- Epochs: 20 (adjustable)
- Device: Automatically uses **GPU if available**, else falls back to CPU.

### 4️⃣ Saving the Model
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler
}, 'fraud_detection_model.pth')
```

---

## 🔍 Model Testing
To test the model on a single transaction sample:

```python
import torch
import numpy as np
import joblib

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("fraud_detection_model.pth", map_location=device)

input_size = 29  # V1–V28 + Amount
model = FraudNet(input_size).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Example sample (normalized)
sample = np.array([
    -1.3598, -0.0727, 2.5363, 1.3781, -0.3383, 0.4623, 0.2395, 0.0986,
    0.3637, 0.0907, -0.5515, -0.6178, -0.9913, -0.3111, 1.4681, -0.4704,
    0.2079, 0.0257, 0.4039, 0.2514, -0.0183, 0.2778, -0.1104, 0.0669,
    0.1285, -0.1891, 0.1335, -0.0210, 149.62
])

sample_scaled = checkpoint['scaler'].transform(sample.reshape(1, -1))
sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    prediction = model(sample_tensor).item()

print("Fraudulent Transaction" if prediction > 0.5 else "Legitimate Transaction")
```

---

## 📊 Evaluation Metrics
Common metrics used:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC**

---

## 🏁 Summary
✅ Dataset: Credit Card Fraud Detection  
✅ Framework: PyTorch  
✅ Balancing: RandomOverSampler  
✅ Scaling: StandardScaler  
✅ Model: Simple Feedforward NN  
✅ Device Support: GPU/CPU auto selection

---

## 📂 Files
- `creditcard.csv` → Dataset  
- `train_model.py` → Model training script  
- `test_model.py` → Model testing script  
- `fraud_detection_model.pth` → Saved model weights  
- `README.md` → Project documentation
