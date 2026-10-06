import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------
# 1. 한글 폰트 설정 (macOS 및 기타 환경 대응)
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'AppleGothic'  # Mac OS 한글 폰트
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

# ---------------------------------------------------------
# 2. 데이터 로드
# ---------------------------------------------------------
train_path = '/Users/2donghak/Desktop/부캠/머신러닝/mldl/data/house/house_train.csv'
test_path = '/Users/2donghak/Desktop/부캠/머신러닝/mldl/data/house/house_test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"[데이터 크기] Train: {train.shape}, Test: {test.shape}")

# 타겟 변수 분리 및 로그 변환 (주택 가격의 왜도 완화)
y = train['SalePrice']
y_log = np.log1p(y)

# Id 및 타겟 제거 후 특성 분리
train_features = train.drop(columns=['Id', 'SalePrice'])
test_features = test.drop(columns=['Id'])

# ---------------------------------------------------------
# 3. 데이터 전처리 (결측치 처리 & 원-핫 인코딩)
# ---------------------------------------------------------
# 일관된 인코딩을 위해 train과 test 결합
all_features = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)

# 결측치 처리: 수치형 -> 중앙값, 범주형 -> 'None'
num_cols = all_features.select_dtypes(include=[np.number]).columns
cat_cols = all_features.select_dtypes(exclude=[np.number]).columns

for col in num_cols:
    all_features[col] = all_features[col].fillna(all_features[col].median())
for col in cat_cols:
    all_features[col] = all_features[col].fillna('None')

# 원-핫 인코딩
all_features = pd.get_dummies(all_features, drop_first=True)

# 다시 Train / Test 분리
X = all_features.iloc[:len(train)]
X_test = all_features.iloc[len(train):]
feature_names = X.columns.tolist()

# 검증 세트 분리 (8:2) - test.csv에는 실제 정답(SalePrice)이 없으므로 검증 세트로 모델 성능 평가
X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)

# 피처 스케일링 (규제 선형 모델에는 표준화가 필수)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. 모델 정의 및 학습 (Ridge, Lasso, ElasticNet)
# ---------------------------------------------------------
models = {
    'Ridge': Ridge(alpha=10.0, random_state=42),
    'Lasso': Lasso(alpha=0.01, max_iter=5000, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42)
}

results = {}
test_predictions = {}

print("\n" + "="*55)
print(f"{'Model':<12} | {'RMSE':<12} | {'MAE':<10} | {'R2 Score':<8} | {'0이 된 계수 수'}")
print("="*55)

for name, model in models.items():
    # 학습
    model.fit(X_train_scaled, y_train)
    
    # 검증 데이터 평가
    preds_val_log = model.predict(X_val_scaled)
    preds_val = np.expm1(preds_val_log)  # 로그 역변환
    actual_val = np.expm1(y_val)
    
    rmse = np.sqrt(mean_squared_error(actual_val, preds_val))
    mae = mean_absolute_error(actual_val, preds_val)
    r2 = r2_score(actual_val, preds_val)
    zero_coefs = np.sum(model.coef_ == 0)
    
    results[name] = {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'coef': model.coef_
    }
    
    print(f"{name:<12} | {rmse:<12.2f} | {mae:<10.2f} | {r2:<8.4f} | {zero_coefs}/{len(model.coef_)}")
    
    # house_test.csv에 대한 예측값 계산 (제출용)
    test_preds_log = model.predict(X_test_scaled)
    test_predictions[name] = np.expm1(test_preds_log)

print("="*55)

# test 예측 결과 예시 (상위 5개)
pred_df = pd.DataFrame({'Id': test['Id']})
for name in models:
    pred_df[f'SalePrice_{name}'] = test_predictions[name]
print("\n[house_test.csv 예측 결과 미리보기 (상위 5건)]")
print(pred_df.head())

# test 예측 결과를 CSV 파일로 저장
pred_df.to_csv('house_test_predictions.csv', index=False)
print("-> test 예측 결과가 'house_test_predictions.csv'로 저장되었습니다.")

# ---------------------------------------------------------
# 5. 회귀계수(Coefficients) 시각화
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=False)
top_n = 15  # 상위 절댓값 기준 상위 N개 피처

for ax, (name, res) in zip(axes, results.items()):
    coef_series = pd.Series(res['coef'], index=feature_names)
    
    # 절댓값 기준 상위 N개 특성 추출
    top_features = coef_series.abs().sort_values(ascending=False).head(top_n).index
    top_coefs = coef_series[top_features].sort_values()
    
    # 양수/음수 색상 분리 (양수: royalblue, 음수: crimson)
    colors = ['crimson' if c < 0 else 'royalblue' for c in top_coefs.values]
    
    bars = ax.barh(top_coefs.index, top_coefs.values, color=colors, alpha=0.85)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title(f"{name} 회귀계수 (Top {top_n})\n(R²: {res['r2']:.4f}, 0인 계수: {np.sum(res['coef'] == 0)}개)", fontsize=13)
    ax.set_xlabel("회귀 계수 값 (스케일링 기준)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)

plt.suptitle("Ridge vs Lasso vs ElasticNet 회귀계수 비교", fontsize=16, fontweight='bold')
plt.tight_layout()

# 그래프 이미지 저장
save_img_path = 'house_coefficients_comparison.png'
plt.savefig(save_img_path, dpi=300, bbox_inches='tight')
print(f"-> 회귀계수 시각화 그래프가 '{save_img_path}'로 저장되었습니다.")

plt.show()
