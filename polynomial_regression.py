import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# 1. 재현성을 위한 시드 설정
np.random.seed(42)

# 2. 1,000개의 샘플 데이터 생성 (X 범위: -3 ~ 3)
n_samples = 1000
X = np.sort(np.random.uniform(-3, 3, size=(n_samples, 1)), axis=0)

# 실제 관계식: y = 0.5 * X^3 - X^2 + 2 * X + 1 + Noise
noise = np.random.normal(0, 2.0, size=(n_samples, 1))
y = 0.5 * (X**3) - (X**2) + 2 * X + 1 + noise

# 3. 다항 특성 변환 (3차 다항식 적용)
degree = 3
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly_features.fit_transform(X)

# 4. 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_poly, y)

# 예측값 계산
y_pred = model.predict(X_poly)

# 모델 평가
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# 5. 그래프 시각화
plt.figure(figsize=(10, 6))

# (1) 샘플 데이터 산점도 (반투명 처리로 1000개 분포 확인 용이)
plt.scatter(X, y, color="royalblue", alpha=0.3, label="Sample Data (n=1000)")

# (2) 회귀선 플롯
plt.plot(X, y_pred, color="crimson", linewidth=2.5, label=f"Polynomial Fit (degree={degree})")

# (3) 원본 실제 함수선 (노이즈 제외)
y_true = 0.5 * (X**3) - (X**2) + 2 * X + 1
plt.plot(X, y_true, color="black", linestyle="--", linewidth=1.5, label="True Function")

plt.title("Polynomial Regression with 1,000 Samples", fontsize=14)
plt.xlabel("X", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

# 그래프 출력
plt.show()
