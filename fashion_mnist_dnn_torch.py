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

# 데이터 로드 + 정규화 (0~255 -> 0~1)
transform = transforms.ToTensor()

train_full = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

# validation_split=0.1 과 동일: 학습 데이터의 10%를 검증용으로 분리
val_size = int(len(train_full) * 0.1)
train_size = len(train_full) - val_size
train_set, val_set = random_split(train_full, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


# DNN 구조 정의: Flatten(784) -> Dense(128, relu) -> Dense(64, relu) -> Dense(10)
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # softmax는 CrossEntropyLoss 내부에서 처리되므로 여기선 생략
        return x


model = DNN().to(device)
print(model)

# 학습 설정 (옵티마이저/손실함수)
criterion = nn.CrossEntropyLoss()  # softmax + cross entropy를 합친 것 (sparse_categorical_crossentropy와 동일 역할)
optimizer = optim.Adam(model.parameters())

epochs = 10

# 에포크 학습 시작
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

# 테스트 데이터 평가
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

# 테스트 데이터 500번째(인덱스 500) 예측
idx = 500
sample, true_label = test_set[idx]

with torch.no_grad():
    logits = model(sample.unsqueeze(0).to(device))
    pred_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
pred_label = int(np.argmax(pred_probs))

print(f"\n[테스트 500번째 데이터]")
print(f"실제 값: {class_names[true_label]}")
print(f"예측 값: {class_names[pred_label]}")
print(f"예측 확률: {pred_probs[pred_label]:.4f}")
print(f"정답 여부: {'정답' if pred_label == true_label else '오답'}")

# 시각화: 이미지 + 예측 확률 분포
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(sample.squeeze(0).numpy(), cmap="gray")
color = "green" if pred_label == true_label else "red"
axes[0].set_title(
    f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
    color=color,
)
axes[0].axis("off")

axes[1].bar(class_names, pred_probs)
axes[1].set_xticklabels(class_names, rotation=45, ha="right")
axes[1].set_ylabel("probability")
axes[1].set_title("predicted class probabilities")

plt.tight_layout()
plt.savefig("output/fashion_mnist_test500_prediction_torch.png")
plt.close()

print("\n결과 이미지 저장: output/fashion_mnist_test500_prediction_torch.png")
