import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 데이터 로드 + 정규화 (0~255 -> 0~1)
transform = transforms.ToTensor()

train_full = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

val_size = int(len(train_full) * 0.1)
train_size = len(train_full) - val_size
train_set, val_set = random_split(train_full, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


# 2. CNN 구조 정의
# Conv(32) -> ReLU -> MaxPool -> Conv(64) -> ReLU -> MaxPool -> Flatten -> Dense(128) -> Dense(10)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # 28x28 -> pool -> 14x14 -> pool -> 7x7, 채널 64개
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # (B, 32, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))  # (B, 64, 7, 7)
        x = x.view(x.size(0), -1)                # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN().to(device)
print(model)

# 3. 학습 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

epochs = 10

# 4. 에포크 학습
for epoch in range(epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    print(
        f"Epoch {epoch + 1}/{epochs} - "
        f"loss: {train_loss / train_total:.4f} - accuracy: {train_correct / train_total:.4f} - "
        f"val_loss: {val_loss / val_total:.4f} - val_accuracy: {val_correct / val_total:.4f}"
    )

# 5. 테스트 데이터 전체 평가
model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * images.size(0)
        test_correct += (outputs.argmax(1) == labels).sum().item()
        test_total += labels.size(0)

test_acc = test_correct / test_total
test_loss = test_loss / test_total
print(f"\n테스트 정확도: {test_acc:.4f}, 테스트 손실: {test_loss:.4f}")

# 6. 테스트 데이터 200번째(인덱스 200) 예측
idx = 200
sample, true_label = test_set[idx]

with torch.no_grad():
    logits = model(sample.unsqueeze(0).to(device))
    pred_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
pred_label = int(np.argmax(pred_probs))

print(f"\n[테스트 200번째 데이터]")
print(f"실제 값: {class_names[true_label]}")
print(f"예측 값: {class_names[pred_label]}")
print(f"예측 확률: {pred_probs[pred_label]:.4f}")
print(f"정답 여부: {'정답' if pred_label == true_label else '오답'}")

# 7. 시각화: 이미지 + 예측 확률 분포
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(sample.squeeze(0).numpy(), cmap="gray")
color = "green" if pred_label == true_label else "red"
axes[0].set_title(
    f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
    color=color,
)
axes[0].axis("off")

axes[1].bar(class_names, pred_probs)
axes[1].set_xticks(range(len(class_names)))
axes[1].set_xticklabels(class_names, rotation=45, ha="right")
axes[1].set_ylabel("probability")
axes[1].set_title("predicted class probabilities")

plt.tight_layout()
plt.savefig("output/fashion_mnist_cnn_test200_prediction.png")
plt.close()

print("\n결과 이미지 저장: output/fashion_mnist_cnn_test200_prediction.png")
