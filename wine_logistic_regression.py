import os
import platform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 0. 한글 폰트 및 시각화 기본 설정
# ---------------------------------------------------------
if platform.system() == "Darwin":  # macOS
    plt.rcParams["font.family"] = "AppleGothic"
elif platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------
# 1. 데이터 로드 및 탐색
# ---------------------------------------------------------
print("=" * 65)
print("1. 와인(Wine) 데이터셋 로드 및 특성 확인")
print("=" * 65)

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="target")
target_names = wine.target_names

print(f"- 샘플 수: {X.shape[0]}개, 특성 수: {X.shape[1]}개")
print(f"- 타깃 클래스: {list(target_names)} (0, 1, 2)")
print("\n[특성별 평균 및 표준편차 - 스케일 차이 확인 (상위 5개 특성)]")
stats = X.describe().T[["mean", "std", "min", "max"]].head(5)
print(stats.to_string())
print("\n-> 예: proline(최대 1680)과 nonflavanoid_phenols(최대 0.66)처럼 단위/스케일 차이가 매우 큼!")

# ---------------------------------------------------------
# 2. 데이터 분할 (Train / Test Split)
# ---------------------------------------------------------
# 다중 클래스이므로 계층적 샘플링(stratify=y) 적용
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\n- Train 샘플 수: {len(X_train)}개, Test 샘플 수: {len(X_test)}개")

# ---------------------------------------------------------
# 3. StandardScaler 전처리 효과 실증 비교
# ---------------------------------------------------------
print("\n" + "=" * 65)
print("2. StandardScaler 전처리 적용 전 vs 적용 후 비교")
print("=" * 65)

# (1) 스케일링 미적용 모델
model_raw = LogisticRegression(max_iter=1000, random_state=42)
model_raw.fit(X_train, y_train)
y_pred_raw = model_raw.predict(X_test)
acc_raw = accuracy_score(y_test, y_pred_raw)

# (2) StandardScaler 적용 모델
model_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])
model_scaled.fit(X_train, y_train)
y_pred_scaled = model_scaled.predict(X_test)
y_prob_scaled = model_scaled.predict_proba(X_test)

acc_scaled = accuracy_score(y_test, y_pred_scaled)
loss_scaled = log_loss(y_test, y_prob_scaled)

print(f"- [스케일링 미적용] Test 정확도: {acc_raw * 100:.2f}% (수렴 불안정 및 특정 피처 편향 발생)")
print(f"- [StandardScaler 적용] Test 정확도: {acc_scaled * 100:.2f}%")
print(f"- [StandardScaler 적용] 다중 클래스 교차 엔트로피 손실(Log-Loss): {loss_scaled:.4f}")

# ---------------------------------------------------------
# 4. 상세 평가: 분류 보고서 및 소프트맥스 예측 확률
# ---------------------------------------------------------
print("\n" + "=" * 65)
print("3. 모델 상세 평가 지표 (Classification Report)")
print("=" * 65)
print(classification_report(y_test, y_pred_scaled, target_names=target_names))

print("\n[소프트맥스(Softmax) 클래스별 예측 확률 샘플 (상위 3개)]")
prob_df = pd.DataFrame(
    y_prob_scaled[:3],
    columns=[f"P({c})" for c in target_names]
)
prob_df["예측 클래스"] = [target_names[p] for p in y_pred_scaled[:3]]
prob_df["실제 클래스"] = [target_names[t] for t in y_test.iloc[:3]]
print(prob_df.to_string(index=False))

# ---------------------------------------------------------
# 5. 시각화 (혼동 행렬 + 클래스별 회귀 계수)
# ---------------------------------------------------------
print("\n" + "=" * 65)
print("4. 결과 시각화 생성 중...")
print("=" * 65)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# (1) 혼동 행렬(Confusion Matrix) 히트맵
cm = confusion_matrix(y_test, y_pred_scaled)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
    ax=axes[0],
    cbar=False,
    annot_kws={"size": 13}
)
axes[0].set_title("혼동 행렬 (Confusion Matrix)", fontsize=14, fontweight="bold")
axes[0].set_xlabel("예측 품종 (Predicted)", fontsize=12)
axes[0].set_ylabel("실제 품종 (Actual)", fontsize=12)

# (2) 클래스별 회귀 계수(Coefficients) 히트맵
# 다중 클래스 로지스틱 회귀는 각 클래스마다 1개씩 선형 판별 함수를 가짐 (shape: [3, 13])
coef = model_scaled.named_steps["clf"].coef_
coef_df = pd.DataFrame(coef, index=target_names, columns=wine.feature_names)

sns.heatmap(
    coef_df,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    ax=axes[1],
    cbar_kws={"label": "회귀 계수 (가중치)"},
    annot_kws={"size": 10}
)
axes[1].set_title("각 와인 품종별 피처 회귀 계수 (중요 특성 분석)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("화학 성분 특성 (Features)", fontsize=12)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=10)

plt.tight_layout()
output_img = "wine_logistic_regression.png"
plt.savefig(output_img, dpi=300, bbox_inches="tight")
plt.close()
print(f"- 시각화 이미지 저장 완료: {output_img}")
