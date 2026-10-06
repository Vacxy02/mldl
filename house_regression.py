import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------
# 0. 한글 폰트 및 시각화 기본 설정
# ---------------------------------------------------------
import platform
if platform.system() == "Darwin":  # macOS
    plt.rcParams["font.family"] = "AppleGothic"
elif platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------
base_path = "/Users/2donghak/Desktop/부캠/머신러닝/mldl/data/house"
train_path = os.path.join(base_path, "house_train.csv")
test_path = os.path.join(base_path, "house_test.csv")
sub_path = os.path.join(base_path, "house_sample_submission.csv")

print("[1/5] 데이터 불러오는 중...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(sub_path)

print(f"- Train 크기: {train.shape}")
print(f"- Test 크기: {test.shape}")

# ---------------------------------------------------------
# 2. 데이터 전처리 파이프라인
# ---------------------------------------------------------
print("\n[2/5] 피처 전처리 진행 중...")
# 타깃 변수와 피처 분리 (SalePrice 로그 변환)
X = train.drop(columns=["Id", "SalePrice"])
y = train["SalePrice"]
y_log = np.log1p(y)  # 정규성 확보를 위한 로그 변환

X_test = test.drop(columns=["Id"])

# 수치형, 범주형 컬럼 분류
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

# 전처리 파이프라인 구성
# 수치형: 결측치 중앙값 대체 + 표준화(StandardScaler)
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 범주형: 결측치 대체 + 원핫인코딩(OneHotEncoder)
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ]
)

# 학습 데이터 및 테스트 데이터 변환
X_prep = preprocessor.fit_transform(X)
X_test_prep = preprocessor.transform(X_test)

# 피처 이름 추출 (회귀 계수 시각화 시 사용)
encoded_cat_cols = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_cols).tolist()
feature_names = num_cols + encoded_cat_cols
print(f"- 전처리 완료된 총 피처 개수: {len(feature_names)}")

# ---------------------------------------------------------
# 3. 모델 정의 및 교차 검증 (릿지, 라쏘, 엘라스틱넷)
# ---------------------------------------------------------
print("\n[3/5] 모델 학습 및 교차 검증(K-Fold CV) 평가...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Ridge": RidgeCV(alphas=np.logspace(-2, 3, 50), cv=cv),
    "Lasso": LassoCV(alphas=np.logspace(-4, 0, 50), cv=cv, max_iter=10000, random_state=42),
    "ElasticNet": ElasticNetCV(alphas=np.logspace(-4, 0, 30), l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                               cv=cv, max_iter=10000, random_state=42)
}

eval_results = []
trained_models = {}

for name, model in models.items():
    # 로그 스케일에서 교차 검증 예측 수행
    y_pred_log = cross_val_predict(model, X_prep, y_log, cv=cv)
    # 실제 가격 스케일로 역변환 (expm1)
    y_pred = np.expm1(y_pred_log)
    
    # 평가 지표 계산 (대회 공식 기준인 RMSLE 및 실거래가 기준 지표)
    rmsle = np.sqrt(mean_squared_error(y_log, y_pred_log))
    r2_log = r2_score(y_log, y_pred_log)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    eval_results.append({
        "Model": name,
        "RMSLE (Log RMSE)": round(rmsle, 4),
        "Log R2 Score": round(r2_log, 4),
        "MAE (실제가격 $)": f"${mae:,.0f}",
        "RMSE (실제가격 $)": f"${rmse:,.0f}"
    })
    
    # 전체 데이터로 최종 모델 피팅
    model.fit(X_prep, y_log)
    trained_models[name] = model

eval_df = pd.DataFrame(eval_results)
print("\n=== [5-Fold Cross Validation 성능 평가 결과] ===")
print(eval_df.to_string(index=False))

# ---------------------------------------------------------
# 4. 테스트 데이터 예측 및 제출 파일(submission) 생성
# ---------------------------------------------------------
print("\n[4/5] Test 데이터셋 예측 및 submission 파일 생성...")
for name, model in trained_models.items():
    # 최적 파라미터 출력
    if hasattr(model, "alpha_"):
        param_info = f"alpha={model.alpha_:.5f}"
        if hasattr(model, "l1_ratio_"):
            param_info += f", l1_ratio={model.l1_ratio_:.2f}"
        print(f"- {name} 최적 파라미터: {param_info}")
    
    # test 데이터 예측 및 expm1 복원
    test_pred_log = model.predict(X_test_prep)
    test_pred = np.expm1(test_pred_log)
    
    sub = submission.copy()
    sub["SalePrice"] = test_pred
    out_file = os.path.join(base_path, f"house_submission_{name.lower()}.csv")
    sub.to_csv(out_file, index=False)
    print(f"  -> 저장 완료: {out_file}")

# ---------------------------------------------------------
# 5. 회귀 계수(Coefficients) 시각화
# ---------------------------------------------------------
print("\n[5/5] 회귀 계수 시각화 차트 생성 중...")
fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=False)
fig.suptitle("Ridge, Lasso, ElasticNet 상위 15개 회귀 계수 (영향력 비교)", fontsize=18, fontweight="bold")

top_n = 15

for idx, (name, model) in enumerate(trained_models.items()):
    coefs = model.coef_
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs,
        "Abs_Coefficient": np.abs(coefs)
    }).sort_values(by="Abs_Coefficient", ascending=False).head(top_n)
    
    # 내림차순 정렬 (차트에서는 위쪽이 큰 값)
    coef_df = coef_df.sort_values(by="Coefficient", ascending=True)
    
    colors = ["#e74c3c" if c > 0 else "#3498db" for c in coef_df["Coefficient"]]
    
    ax = axes[idx]
    bars = ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="black", alpha=0.85)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{name} (비영 계수 수: {np.sum(coefs != 0)}/{len(coefs)})", fontsize=14, fontweight="bold")
    ax.set_xlabel("회귀 계수 (log 스케일 기여도)", fontsize=12)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    
    # 수치 라벨 추가
    for bar in bars:
        width = bar.get_width()
        ha = "left" if width >= 0 else "right"
        offset = 0.003 if width >= 0 else -0.003
        ax.text(width + offset, bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}", va="center", ha=ha, fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
output_chart = os.path.join(base_path, "house_regression_coefficients.png")
plt.savefig(output_chart, dpi=300, bbox_inches="tight")
print(f"- 시각화 그래프 저장 완료: {output_chart}")
plt.close()

print("\n모든 작업이 성공적으로 완료되었습니다!")
