import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# 1. 이미지 불러오기
image_path = "Gemini_Generated_Image_22i6122i6122i612.png"
img = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # (C, H, W), 0~1 정규화
])
img_tensor = transform(img).unsqueeze(0)  # (1, 3, 128, 128) 배치 차원 추가

# 2. 필터 10개짜리 convolution 레이어 정의
# in_channels=3 (RGB), out_channels=10 (필터 10개), kernel_size=3
conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)

# 3. convolution 연산 수행
with torch.no_grad():
    output = conv(img_tensor)  # (1, 10, 128, 128)

print("입력 이미지 shape:", img_tensor.shape)
print("출력 feature map shape:", output.shape)  # 필터 10개 -> 채널 10개로 출력

# 4. 원본 이미지 + 10개 필터 결과(feature map) 시각화
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

axes[0].imshow(img_tensor[0].permute(1, 2, 0).numpy())
axes[0].set_title("Original")
axes[0].axis("off")

for i in range(10):
    feature_map = output[0, i].numpy()
    axes[i + 1].imshow(feature_map, cmap="viridis")
    axes[i + 1].set_title(f"Filter {i + 1}")
    axes[i + 1].axis("off")

for j in range(11, 12):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig("output/conv_filter10_result.png", dpi=150)
plt.close()

print("\n결과 이미지 저장: output/conv_filter10_result.png")
