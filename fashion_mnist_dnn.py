import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# 정규화 (0~255 -> 0~1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 모델 구성: Flatten(784) -> Dense(128, relu) -> Dense(64, relu) -> Dense(10, softmax)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n테스트 정확도: {test_acc:.4f}, 테스트 손실: {test_loss:.4f}")

# 테스트 데이터 500번째(인덱스 500) 예측
idx = 500
sample = x_test[idx]
true_label = y_test[idx]

pred_probs = model.predict(sample[np.newaxis, ...], verbose=0)[0]
pred_label = np.argmax(pred_probs)

print(f"\n[테스트 500번째 데이터]")
print(f"실제 값: {class_names[true_label]}")
print(f"예측 값: {class_names[pred_label]}")
print(f"예측 확률: {pred_probs[pred_label]:.4f}")
print(f"정답 여부: {'정답' if pred_label == true_label else '오답'}")

# 시각화: 이미지 + 예측 확률 분포
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(sample, cmap="gray")
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
plt.savefig("output/fashion_mnist_test500_prediction.png")
plt.close()

print("\n결과 이미지 저장: output/fashion_mnist_test500_prediction.png")
