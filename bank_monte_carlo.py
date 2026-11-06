import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('output', exist_ok=True)

# ── 1. 데이터 로드 & 전처리 ──────────────────────────────────────
df = pd.read_csv('data/bank-full.csv', sep=';')

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

# ── 2. 모델 학습 ─────────────────────────────────────────────────
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, random_state=42, verbosity=0
)
model.fit(X_train, y_train)
print("모델 학습 완료\n")

# ── 3. 피처 분포 추정 (수치형 → 정규분포, 범주형 → 이산분포) ──────
num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'day']
cat_cols  = [c for c in X.columns if c not in num_cols]

num_stats = X_train[num_cols].agg(['mean', 'std'])
cat_probs  = {col: X_train[col].value_counts(normalize=True) for col in cat_cols}

N_SIM = 10_000
np.random.seed(42)

# ── 4. 몬테카를로 시뮬레이션 ─────────────────────────────────────
print(f"몬테카를로 시뮬레이션 {N_SIM:,}회 실행 중...")

sim_data = {}

# 수치형: 정규분포에서 샘플링 (음수 방지 위해 clip)
for col in num_cols:
    mu, sigma = num_stats[col]['mean'], num_stats[col]['std']
    samples = np.random.normal(mu, sigma, N_SIM)
    # 원래 범위로 clip
    lo, hi = X_train[col].min(), X_train[col].max()
    sim_data[col] = np.clip(samples, lo, hi)

# 범주형: 실제 비율로 샘플링
for col in cat_cols:
    probs = cat_probs[col]
    sim_data[col] = np.random.choice(probs.index, size=N_SIM, p=probs.values)

X_sim = pd.DataFrame(sim_data)[X.columns]
y_prob_sim = model.predict_proba(X_sim)[:, 1]

print(f"시뮬레이션 완료\n")
print("=== 예측 확률 분포 ===")
print(f"  평균: {y_prob_sim.mean():.4f}")
print(f"  표준편차: {y_prob_sim.std():.4f}")
print(f"  5%ile: {np.percentile(y_prob_sim, 5):.4f}")
print(f"  95%ile: {np.percentile(y_prob_sim, 95):.4f}")
print(f"  P(yes > 0.5): {(y_prob_sim > 0.5).mean():.4f} ({(y_prob_sim > 0.5).mean()*100:.1f}%)\n")

# ── 5. 피처별 몬테카를로 민감도 (One-At-A-Time) ───────────────────
# 각 피처를 랜덤 변화시키고 예측 분산 기여도 측정
print("피처별 민감도 계산 중...")

X_base_sim = X_sim.copy()
base_probs  = model.predict_proba(X_base_sim)[:, 1]

sensitivity = {}
for col in num_cols:
    X_perturbed = X_base_sim.copy()
    noise = np.random.normal(0, num_stats[col]['std'] * 0.1, N_SIM)
    X_perturbed[col] = np.clip(X_base_sim[col] + noise,
                                X_train[col].min(), X_train[col].max())
    perturbed_probs = model.predict_proba(X_perturbed)[:, 1]
    sensitivity[col] = np.std(perturbed_probs - base_probs)

sensitivity_df = pd.Series(sensitivity).sort_values(ascending=False)
print("\n=== 피처별 몬테카를로 민감도 (표준편차 기준) ===")
for feat, val in sensitivity_df.items():
    print(f"  {feat:<12s}  σ={val:.5f}")

# ── 6. 임계값별 가입 확률 분포 (VaR 스타일) ──────────────────────
thresholds = np.arange(0.1, 1.0, 0.05)
exceed_probs = [(y_prob_sim > t).mean() for t in thresholds]

# ── 7. 시각화 ────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))

# (1) 시뮬레이션 예측 확률 히스토그램
ax1 = fig.add_subplot(3, 3, 1)
ax1.hist(y_prob_sim, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
ax1.axvline(y_prob_sim.mean(), color='red', lw=2, label=f'평균={y_prob_sim.mean():.3f}')
ax1.axvline(0.5, color='orange', lw=1.5, linestyle='--', label='임계값=0.5')
ax1.set_xlabel('예측 확률 (P(yes))')
ax1.set_ylabel('빈도')
ax1.set_title(f'몬테카를로 예측 분포 (N={N_SIM:,})')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# (2) 누적 분포 (CDF)
ax2 = fig.add_subplot(3, 3, 2)
sorted_probs = np.sort(y_prob_sim)
cdf = np.arange(1, N_SIM + 1) / N_SIM
ax2.plot(sorted_probs, cdf, color='steelblue', lw=2)
ax2.axvline(0.5, color='orange', lw=1.5, linestyle='--', label='임계값=0.5')
ax2.fill_betweenx([0, 1], np.percentile(y_prob_sim, 5),
                   np.percentile(y_prob_sim, 95),
                   alpha=0.1, color='steelblue', label='90% 신뢰구간')
ax2.set_xlabel('예측 확률')
ax2.set_ylabel('누적 확률')
ax2.set_title('누적 분포 (CDF)')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# (3) 피처별 민감도 막대
ax3 = fig.add_subplot(3, 3, 3)
sensitivity_df.plot(kind='barh', ax=ax3, color='coral', edgecolor='white')
ax3.set_xlabel('예측 변화 표준편차 (σ)')
ax3.set_title('피처별 몬테카를로 민감도')
ax3.grid(alpha=0.3, axis='x')

# (4) 임계값별 가입 예측 비율
ax4 = fig.add_subplot(3, 3, 4)
ax4.plot(thresholds, exceed_probs, 'o-', color='steelblue', lw=2)
ax4.set_xlabel('결정 임계값')
ax4.set_ylabel('가입 예측 비율')
ax4.set_title('임계값별 예측 가입률')
ax4.grid(alpha=0.3)

# (5~) 수치형 피처 시뮬레이션 분포 vs 실제 분포
for idx, col in enumerate(num_cols[:4]):
    ax = fig.add_subplot(3, 3, 5 + idx)
    ax.hist(X_train[col], bins=40, alpha=0.5, color='gray', label='실제 데이터', density=True)
    ax.hist(X_sim[col],   bins=40, alpha=0.5, color='steelblue', label='MC 샘플', density=True)
    ax.set_xlabel(col)
    ax.set_ylabel('밀도')
    ax.set_title(f'{col} 분포 비교')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Bank Marketing — 몬테카를로 민감도 분석', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig('output/monte_carlo_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n저장: output/monte_carlo_analysis.png")

# ── 8. 시나리오 분석 ─────────────────────────────────────────────
# duration이 가장 중요 → duration 구간별 가입 확률 분포
print("\n=== 시나리오 분석: duration 구간별 평균 가입 확률 ===")
bins = [0, 60, 120, 300, 600, 1200, 9999]
labels = ['~1분', '1~2분', '2~5분', '5~10분', '10~20분', '20분+']
X_sim['y_prob'] = y_prob_sim
X_sim['duration_bin'] = pd.cut(X_sim['duration'], bins=bins, labels=labels)
scenario = X_sim.groupby('duration_bin', observed=True)['y_prob'].agg(['mean', 'std', 'count'])
print(scenario.to_string())

fig2, ax = plt.subplots(figsize=(9, 5))
ax.bar(scenario.index.astype(str), scenario['mean'],
       yerr=scenario['std'], capsize=5,
       color='steelblue', edgecolor='white', alpha=0.85)
ax.set_xlabel('통화 시간 구간')
ax.set_ylabel('평균 가입 예측 확률')
ax.set_title('통화 시간 시나리오별 예측 가입 확률 (± std)')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('output/monte_carlo_scenario.png', dpi=150)
plt.close()
print("저장: output/monte_carlo_scenario.png")
