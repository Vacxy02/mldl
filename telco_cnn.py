import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class_names = ["No Churn", "Churn"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 전처리된 데이터 불러오기
df = pd.read_csv("data/telco_churn_preprocessed.csv")

X = df.drop(columns=["Churn"]).values.astype("float32")  # (7043, 19)
y = df["Churn"].values.astype("int64")

# 2. train / test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 텐서 변환: 1D-CNN 입력 형태 (배치, 채널=1, 길이=피처개수)
X_train_t = torch.tensor(X_train).unsqueeze(1)  # (N, 1, 19)
X_test_t = torch.tensor(X_test).unsqueeze(1)
y_train_t = torch.tensor(y_train)
y_test_t = torch.tensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)

n_features = X.shape[1]


# 4. 1D-CNN 구조 정의
# Conv1d(16) -> ReLU -> MaxPool -> Conv1d(32) -> ReLU -> Flatten -> Dense(32) -> Dense(2)
class CNN1D(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()

        pooled_len = n_features // 2  # conv1 -> pool 한 번만 적용
        self.fc1 = nn.Linear(32 * pooled_len, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))   # (B, 16, 19)
        x = self.pool(x)               # (B, 16, 9)
        x = self.relu(self.conv2(x))   # (B, 32, 9)
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN1D(n_features).to(device)
print(model)

# 5. 학습 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

epochs = 20

# 6. 에포크 학습
for epoch in range(epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * features.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    print(
        f"Epoch {epoch + 1}/{epochs} - "
        f"loss: {train_loss / train_total:.4f} - accuracy: {train_correct / train_total:.4f}"
    )

# 7. 테스트 데이터 전체 평가
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        test_correct += (outputs.argmax(1) == labels).sum().item()
        test_total += labels.size(0)

print(f"\n테스트 정확도: {test_correct / test_total:.4f}")

# 8. 테스트 데이터 중 하나 예측 (인덱스 지정)
idx = 100
sample = X_test_t[idx]
true_label = int(y_test_t[idx])

with torch.no_grad():
    logits = model(sample.unsqueeze(0).to(device))
    pred_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
pred_label = int(np.argmax(pred_probs))

print(f"\n[테스트 {idx}번째 데이터]")
print(f"실제 값: {class_names[true_label]}")
print(f"예측 값: {class_names[pred_label]}")
print(f"예측 확률: {pred_probs[pred_label]:.4f}")
print(f"정답 여부: {'정답' if pred_label == true_label else '오답'}")

# 9. 시각화: 입력 피처 값 + 예측 확률
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

feature_names = df.drop(columns=["Churn"]).columns.tolist()
axes[0].barh(feature_names, sample.squeeze(0).numpy())
axes[0].set_title(f"Input features (idx={idx})")

color = "green" if pred_label == true_label else "red"
axes[1].bar(class_names, pred_probs, color=["steelblue", "orange"])
axes[1].set_ylim(0, 1)
axes[1].set_title(
    f"True: {class_names[true_label]} / Pred: {class_names[pred_label]}",
    color=color,
)
axes[1].set_ylabel("probability")

plt.tight_layout()
plt.savefig("output/telco_cnn_prediction.png", dpi=150)
plt.close()

print("\n결과 이미지 저장: output/telco_cnn_prediction.png")
