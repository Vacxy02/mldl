import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

# ── 1. 데이터 로드 ──────────────────────────────────────────────
df = pd.read_csv('data/bank-full.csv', sep=';')
print(f"데이터 shape: {df.shape}")
print(f"타겟 분포:\n{df['y'].value_counts()}\n")

# ── 2. 전처리 ────────────────────────────────────────────────────
df_enc = df.copy()
label_cols = ['job', 'marital', 'education', 'default',
              'housing', 'loan', 'contact', 'month', 'poutcome']

le = LabelEncoder()
for col in label_cols:
    df_enc[col] = le.fit_transform(df_enc[col])

df_enc['y'] = (df_enc['y'] == 'yes').astype(int)

X = df_enc.drop('y', axis=1)
y = df_enc['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. XGBoost 모델 학습 ─────────────────────────────────────────
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,  # 클래스 불균형 처리
    eval_metric='auc',
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=== 모델 성능 ===")
print(classification_report(y_test, y_pred, target_names=['no', 'yes']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}\n")

# ── 4. SHAP 민감도 분석 ──────────────────────────────────────────
print("SHAP 값 계산 중...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

os_path = 'output'
import os; os.makedirs(os_path, exist_ok=True)

# (1) SHAP Summary Plot — 전체 피처 중요도
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP 피처 중요도 (전체 평균)", fontsize=14)
plt.tight_layout()
plt.savefig(f'{os_path}/shap_importance_bar.png', dpi=150)
plt.close()

# (2) SHAP Beeswarm Plot — 방향성 포함
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Beeswarm — 피처별 영향 방향", fontsize=14)
plt.tight_layout()
plt.savefig(f'{os_path}/shap_beeswarm.png', dpi=150)
plt.close()

# (3) 수동 민감도 분석 — 수치형 피처별 변화에 따른 예측 변화
num_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'day']
X_base = X_test.mean().to_frame().T  # 평균값 기준점

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, feat in enumerate(num_features):
    feat_range = np.linspace(X_test[feat].quantile(0.05),
                             X_test[feat].quantile(0.95), 50)
    preds = []
    for val in feat_range:
        X_tmp = X_base.copy()
        X_tmp[feat] = val
        prob = model.predict_proba(X_tmp)[0, 1]
        preds.append(prob)

    axes[i].plot(feat_range, preds, color='steelblue', linewidth=2)
    axes[i].axhline(y=model.predict_proba(X_base)[0, 1],
                    color='red', linestyle='--', alpha=0.6, label='기준값')
    axes[i].set_xlabel(feat, fontsize=11)
    axes[i].set_ylabel('예측 확률 (yes)', fontsize=10)
    axes[i].set_title(f'{feat} 민감도', fontsize=12)
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.3)

# 빈 subplot 제거
for j in range(len(num_features), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("수치형 피처 민감도 분석 (다른 피처 = 평균값 고정)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f'{os_path}/sensitivity_numeric.png', dpi=150, bbox_inches='tight')
plt.close()

# (4) SHAP Dependence Plot — 가장 중요한 피처 2개
top2 = pd.Series(
    np.abs(shap_values).mean(axis=0), index=X_test.columns
).nlargest(2).index.tolist()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, feat in zip(axes, top2):
    shap.dependence_plot(feat, shap_values, X_test, ax=ax, show=False)
    ax.set_title(f'SHAP Dependence: {feat}', fontsize=12)

plt.tight_layout()
plt.savefig(f'{os_path}/shap_dependence_top2.png', dpi=150)
plt.close()

# ── 5. 결과 요약 출력 ────────────────────────────────────────────
mean_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X_test.columns)
mean_shap = mean_shap.sort_values(ascending=False)

print("=== SHAP 피처 중요도 순위 ===")
for rank, (feat, val) in enumerate(mean_shap.items(), 1):
    print(f"  {rank:2d}. {feat:<12s}  SHAP={val:.4f}")

print(f"\n저장된 파일:")
print(f"  output/shap_importance_bar.png  — 피처 중요도 막대그래프")
print(f"  output/shap_beeswarm.png        — 방향성 포함 Beeswarm")
print(f"  output/sensitivity_numeric.png  — 수치형 피처 민감도 곡선")
print(f"  output/shap_dependence_top2.png — 상위 2개 피처 Dependence Plot")
