import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# 0. 한글 폰트 및 시각화 스타일 설정
# -------------------------------------------------------------
plt.rcParams['font.family'] = 'AppleGothic'  # Mac 환경 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', font='AppleGothic')

OUTPUT_DIR = 'eda_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. 데이터 로드 및 기본 구조 확인
# -------------------------------------------------------------
train_path = 'mobile_price_category_train.csv'
test_path = 'mobile_price_category_test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("=" * 60)
print("1. DATA OVERVIEW (데이터 개요)")
print("=" * 60)
print(f"Train Shape: {train.shape} (행: {train.shape[0]}, 열: {train.shape[1]})")
print(f"Test Shape : {test.shape}  (행: {test.shape[0]}, 열: {test.shape[1]})")
print("\n[Train Data Info]")
train.info()

print("\n[Train Head (상위 3행)]")
print(train.head(3))

# -------------------------------------------------------------
# 2. 결측치 및 중복값 확인
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("2. MISSING & DUPLICATE VALUES (결측치 및 중복값)")
print("=" * 60)
train_null = train.isnull().sum()
test_null = test.isnull().sum()
print("Train 결측치 총합:", train_null.sum())
print("Test  결측치 총합:", test_null.sum())
print(f"Train 중복행 수 : {train.duplicated().sum()}")
print(f"Test  중복행 수 : {test.duplicated().sum()}")

# -------------------------------------------------------------
# 3. 타깃 변수 (price_range) 분포 확인
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("3. TARGET DISTRIBUTION (타깃 변수 분포)")
print("=" * 60)
target_counts = train['price_range'].value_counts().sort_index()
print("price_range 클래스별 빈도수:")
print(target_counts)

# 시각화: 타깃 변수 분포 플롯
plt.figure(figsize=(7, 4))
ax = sns.countplot(x='price_range', hue='price_range', data=train, palette='viridis', legend=False)
plt.title('타깃 변수(Price Range) 빈도 분포', fontsize=14, pad=12)
plt.xlabel('Price Range (0: 저가 ~ 3: 최고가)', fontsize=11)
plt.ylabel('개수 (Count)', fontsize=11)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_target_distribution.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# 4. 수치형 피처 요약 통계 및 이진 피처 분류
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("4. FEATURE CLASSIFICATION & DESCRIPTIVE STATS (기술 통계)")
print("=" * 60)

# 0과 1로만 이루어진 이진형(Binary) 피처 구분
binary_cols = [c for c in train.columns if train[c].nunique() == 2]
numerical_cols = [c for c in train.columns if c not in binary_cols and c != 'price_range']

print(f"이진형(Binary) 피처 ({len(binary_cols)}개): {binary_cols}")
print(f"연속/수치형 피처    ({len(numerical_cols)}개): {numerical_cols}")

print("\n[수치형 피처 요약 통계]")
print(train[numerical_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])

# -------------------------------------------------------------
# 5. 상관관계 분석 (Correlation Heatmap)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("5. CORRELATION ANALYSIS (상관관계 분석)")
print("=" * 60)
corr = train.corr()
target_corr = corr['price_range'].sort_values(ascending=False)
print("[price_range 와의 상관계수 (높은 순서)]")
print(target_corr)

# 시각화: 전체 상관계수 히트맵
plt.figure(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            cbar=True, square=True, linewidths=0.5, annot_kws={"size": 8})
plt.title('전체 피처 상관계수 히트맵 (Correlation Heatmap)', fontsize=15, pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_correlation_heatmap.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# 6. 핵심 피처와 타깃의 관계 시각화 (Boxplot)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("6. KEY FEATURES VS TARGET VISUALIZATION")
print("=" * 60)
# 상관관계 높은 상위 수치형 피처 4개 선정 (ram, battery_power, px_width, px_height)
top_features = ['ram', 'battery_power', 'px_width', 'px_height']

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

for i, col in enumerate(top_features):
    sns.boxplot(x='price_range', y=col, hue='price_range', data=train, ax=axes[i], palette='Set2', legend=False)
    axes[i].set_title(f'{col} vs Price Range', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Price Range')
    axes[i].set_ylabel(col)

plt.suptitle('핵심 피처와 가격 범주(Price Range) 비교', fontsize=15, y=0.99)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_top_features_vs_target.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# 7. Train vs Test 데이터셋 분포 비교 (KDE Plot)
# -------------------------------------------------------------
print("7. TRAIN VS TEST DISTRIBUTION COMPARISON")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for i, col in enumerate(top_features):
    sns.kdeplot(train[col], label='Train', ax=axes[i], color='royalblue', fill=True, alpha=0.3)
    sns.kdeplot(test[col], label='Test', ax=axes[i], color='coral', fill=True, alpha=0.3)
    axes[i].set_title(f'Train vs Test: {col}', fontsize=12, fontweight='bold')
    axes[i].legend()

plt.suptitle('Train 데이터 vs Test 데이터 주요 피처 분포 비교', fontsize=15, y=0.99)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_train_vs_test_distribution.png', dpi=300)
plt.close()

print(f"\n>> EDA 분석 및 시각화 완료! 차트가 '{OUTPUT_DIR}/' 폴더에 저장되었습니다.")
