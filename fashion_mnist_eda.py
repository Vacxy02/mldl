import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
print("pixel value range:", x_train.min(), "~", x_train.max())

unique, counts = np.unique(y_train, return_counts=True)
print("\n클래스별 학습 데이터 개수")
for u, c in zip(unique, counts):
    print(f"  {u} ({class_names[u]}): {c}")

# 클래스 분포 시각화
plt.figure(figsize=(8, 4))
plt.bar([class_names[u] for u in unique], counts)
plt.xticks(rotation=45, ha="right")
plt.ylabel("count")
plt.title("Fashion MNIST train label distribution")
plt.tight_layout()
plt.savefig("output/fashion_mnist_label_distribution.png")
plt.close()

# 샘플 이미지 여러 개 둘러보기 (클래스별 1개)
plt.figure(figsize=(10, 4))
for i in range(10):
    idx = np.where(y_train == i)[0][0]
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[idx], cmap="gray")
    plt.title(class_names[i])
    plt.axis("off")
plt.tight_layout()
plt.savefig("output/fashion_mnist_sample_per_class.png")
plt.close()

# 1000번 데이터 시각화
idx = 1000
img = x_train[idx]
label = y_train[idx]

print(f"\n{idx}번 데이터 라벨: {label} ({class_names[label]})")

plt.figure(figsize=(4, 4))
plt.imshow(img, cmap="gray")
plt.title(f"index {idx}: {class_names[label]} (label={label})")
plt.axis("off")
plt.savefig("output/fashion_mnist_index_1000.png")
plt.close()
