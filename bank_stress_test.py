"""
Bank Marketing — 스트레스 테스트 비교
  방법 1: Tabular Diffusion Model (DDPM)
  방법 2: Monte Carlo Tail Sampling
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# xgboost를 torch보다 먼저 import (충돌 방지)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

os.makedirs('output', exist_ok=True)
DEVICE = 'cpu'
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# 1. 데이터 로드 & 전처리
# ═══════════════════════════════════════════════════════════════
df = pd.read_csv('data/bank-full.csv', sep=';')

df_enc = df.copy()
label_cols = ['job','marital','education','default',
              'housing','loan','contact','month','poutcome']
le = LabelEncoder()
for col in label_cols:
    df_enc[col] = le.fit_transform(df_enc[col])
df_enc['y'] = (df_enc['y'] == 'yes').astype(int)

X = df_enc.drop('y', axis=1)
y = df_enc['y']

NUM_COLS = ['age','balance','duration','campaign','pdays','previous','day']
CAT_COLS = [c for c in X.columns if c not in NUM_COLS]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 수치형 피처 정규화 (Diffusion에 사용)
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[NUM_COLS])
X_test_num  = scaler.transform(X_test[NUM_COLS])

# ═══════════════════════════════════════════════════════════════
# 2. XGBoost 분류기 학습 (스트레스 시나리오 평가용)
# ═══════════════════════════════════════════════════════════════
print("\n[1/4] XGBoost 학습...")
spw = (y_train == 0).sum() / (y_train == 1).sum()
clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                    scale_pos_weight=spw, random_state=42, verbosity=0)
clf.fit(X_train, y_train)
y_prob_test = clf.predict_proba(X_test)[:, 1]
print(f"  Test ROC-AUC: {roc_auc_score(y_test, y_prob_test):.4f}")

# ═══════════════════════════════════════════════════════════════
# 3. Tabular DDPM 구현
# ═══════════════════════════════════════════════════════════════
class SinusoidalPE(nn.Module):
    """타임스텝 임베딩"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb  = t[:, None].float() * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class TabularDenoiser(nn.Module):
    """수치형 피처용 MLP 디노이저"""
    def __init__(self, n_feat, t_dim=32, h_dim=256):
        super().__init__()
        self.pe  = SinusoidalPE(t_dim)
        self.net = nn.Sequential(
            nn.Linear(n_feat + t_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, h_dim),          nn.SiLU(),
            nn.Linear(h_dim, h_dim),          nn.SiLU(),
            nn.Linear(h_dim, n_feat)
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, self.pe(t)], dim=-1))

class DDPM:
    def __init__(self, n_feat, T=300, device='cpu'):
        self.T = T
        self.device = device
        betas = torch.linspace(1e-4, 0.02, T, device=device)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.ac  = ac
        self.sqrt_ac      = ac.sqrt()
        self.sqrt_1m_ac   = (1 - ac).sqrt()
        self.model = TabularDenoiser(n_feat).to(device)

    def q_sample(self, x0, t):
        noise = torch.randn_like(x0)
        return self.sqrt_ac[t, None] * x0 + self.sqrt_1m_ac[t, None] * noise, noise

    def loss(self, x0):
        bsz = x0.size(0)
        t = torch.randint(0, self.T, (bsz,), device=self.device)
        xt, noise = self.q_sample(x0, t)
        pred = self.model(xt, t)
        return nn.functional.mse_loss(pred, noise)

    @torch.no_grad()
    def generate(self, n, stress_scale=1.0):
        """stress_scale > 1: 극단적 초기 노이즈 → 스트레스 시나리오"""
        n_feat = self.model.net[0].in_features - self.model.pe.dim
        x = torch.randn(n, n_feat, device=self.device) * stress_scale
        for t_val in reversed(range(self.T)):
            t = torch.full((n,), t_val, device=self.device, dtype=torch.long)
            pred_noise  = self.model(x, t)
            beta_t      = self.betas[t_val]
            sqrt_recip  = (1.0 / self.alphas[t_val]).sqrt()
            coef        = beta_t / self.sqrt_1m_ac[t_val]
            mean = sqrt_recip * (x - coef * pred_noise)
            if t_val > 0:
                x = mean + beta_t.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.cpu().numpy()

# ═══════════════════════════════════════════════════════════════
# 4. DDPM 학습
# ═══════════════════════════════════════════════════════════════
print("\n[2/4] Diffusion 모델 학습...")
N_FEAT = len(NUM_COLS)
ddpm = DDPM(N_FEAT, T=100, device=DEVICE)
optimizer = optim.Adam(ddpm.model.parameters(), lr=2e-3)

X_tensor = torch.tensor(X_train_num, dtype=torch.float32).to(DEVICE)
loader   = DataLoader(TensorDataset(X_tensor), batch_size=1024, shuffle=True)

EPOCHS = 40
losses = []
for ep in range(1, EPOCHS + 1):
    ep_loss = 0
    for (xb,) in loader:
        optimizer.zero_grad()
        loss = ddpm.loss(xb)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
    losses.append(ep_loss / len(loader))
    if ep % 20 == 0:
        print(f"  Epoch {ep:3d}/{EPOCHS}  loss={losses[-1]:.4f}")

# ═══════════════════════════════════════════════════════════════
# 5. 스트레스 시나리오 생성
# ═══════════════════════════════════════════════════════════════
print("\n[3/4] 스트레스 시나리오 생성...")
N_GEN = 3000

def build_full_df(num_samples_norm, label='generated'):
    """수치형 + 범주형(실제 비율)을 합쳐 전체 피처 DataFrame 반환"""
    num_df = pd.DataFrame(
        scaler.inverse_transform(num_samples_norm), columns=NUM_COLS)
    # 범주형: 훈련 분포에서 샘플링
    for col in CAT_COLS:
        probs = X_train[col].value_counts(normalize=True)
        num_df[col] = np.random.choice(probs.index, size=len(num_df), p=probs.values)
    return num_df[X.columns]   # 원래 컬럼 순서

# ─── Diffusion 시나리오 ───────────────────────────────────────
diff_normal  = ddpm.generate(N_GEN, stress_scale=1.0)   # 정상
diff_stress1 = ddpm.generate(N_GEN, stress_scale=2.0)   # 중간 스트레스
diff_stress2 = ddpm.generate(N_GEN, stress_scale=3.5)   # 극단 스트레스

df_diff_normal  = build_full_df(diff_normal,  'diff_normal')
df_diff_stress1 = build_full_df(diff_stress1, 'diff_stress1')
df_diff_stress2 = build_full_df(diff_stress2, 'diff_stress2')

# ─── Monte Carlo 시나리오 ─────────────────────────────────────
def mc_sample(n, tail_pct=None):
    """tail_pct=None → 정규분포, tail_pct=0.1 → 하위10%·상위90% 극단값"""
    samples = {}
    for col in NUM_COLS:
        mu, sigma = X_train[col].mean(), X_train[col].std()
        if tail_pct is None:
            vals = np.random.normal(mu, sigma, n)
        else:
            # 하위·상위 tail_pct 범위에서 균등 샘플링
            lo = X_train[col].quantile(tail_pct)
            hi = X_train[col].quantile(1 - tail_pct)
            lower = np.random.uniform(X_train[col].min(), lo, n // 2)
            upper = np.random.uniform(hi, X_train[col].max(), n - n // 2)
            vals  = np.concatenate([lower, upper])
            np.random.shuffle(vals)
        samples[col] = np.clip(vals, X_train[col].min(), X_train[col].max())
    df_mc = pd.DataFrame(samples)
    for col in CAT_COLS:
        probs = X_train[col].value_counts(normalize=True)
        df_mc[col] = np.random.choice(probs.index, size=n, p=probs.values)
    return df_mc[X.columns]

np.random.seed(42)
df_mc_normal  = mc_sample(N_GEN, tail_pct=None)
df_mc_stress1 = mc_sample(N_GEN, tail_pct=0.15)
df_mc_stress2 = mc_sample(N_GEN, tail_pct=0.05)

# ─── 예측 확률 계산 ───────────────────────────────────────────
scenarios = {
    'Diffusion — Normal':   df_diff_normal,
    'Diffusion — Stress1':  df_diff_stress1,
    'Diffusion — Stress2':  df_diff_stress2,
    'MC — Normal':          df_mc_normal,
    'MC — Stress1':         df_mc_stress1,
    'MC — Stress2':         df_mc_stress2,
}
probs = {}
for name, df_s in scenarios.items():
    probs[name] = clf.predict_proba(df_s)[:, 1]

# ═══════════════════════════════════════════════════════════════
# 6. 결과 해석 출력
# ═══════════════════════════════════════════════════════════════
print("\n[4/4] 결과 해석")
print("=" * 65)
print(f"{'시나리오':<25} {'평균':>7} {'표준편차':>8} {'P>0.5':>7} {'5%ile':>7} {'95%ile':>7}")
print("-" * 65)
for name, p in probs.items():
    print(f"{name:<25} {p.mean():>7.4f} {p.std():>8.4f} "
          f"{(p>0.5).mean():>7.3f} {np.percentile(p,5):>7.4f} {np.percentile(p,95):>7.4f}")
print("=" * 65)

# 기준 (실제 테스트셋) 추가
print(f"\n실제 테스트셋 기준:")
print(f"  P(yes) 평균={y_prob_test.mean():.4f}  std={y_prob_test.std():.4f}  P>0.5={(y_prob_test>0.5).mean():.3f}")

# ═══════════════════════════════════════════════════════════════
# 7. 시각화
# ═══════════════════════════════════════════════════════════════
COLORS = {
    'Diffusion — Normal':  '#2196F3',
    'Diffusion — Stress1': '#FF9800',
    'Diffusion — Stress2': '#F44336',
    'MC — Normal':         '#4CAF50',
    'MC — Stress1':        '#FF5722',
    'MC — Stress2':        '#9C27B0',
}

fig, axes = plt.subplots(2, 3, figsize=(17, 10))

# (1) 예측 확률 분포 비교 — Diffusion
ax = axes[0, 0]
for name in ['Diffusion — Normal','Diffusion — Stress1','Diffusion — Stress2']:
    ax.hist(probs[name], bins=50, alpha=0.55, density=True,
            color=COLORS[name], label=name, edgecolor='none')
ax.axvline(0.5, color='black', lw=1.2, linestyle='--')
ax.set_title('Diffusion 스트레스 레벨별 예측 분포')
ax.set_xlabel('P(가입)')
ax.set_ylabel('밀도')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (2) 예측 확률 분포 비교 — MC
ax = axes[0, 1]
for name in ['MC — Normal','MC — Stress1','MC — Stress2']:
    ax.hist(probs[name], bins=50, alpha=0.55, density=True,
            color=COLORS[name], label=name, edgecolor='none')
ax.axvline(0.5, color='black', lw=1.2, linestyle='--')
ax.set_title('Monte Carlo 스트레스 레벨별 예측 분포')
ax.set_xlabel('P(가입)')
ax.set_ylabel('밀도')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (3) 평균±std 비교 막대그래프
ax = axes[0, 2]
names = list(probs.keys())
means = [probs[n].mean() for n in names]
stds  = [probs[n].std()  for n in names]
colors_list = [COLORS[n] for n in names]
bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5,
              color=colors_list, edgecolor='white', alpha=0.85)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace(' — ', '\n') for n in names], fontsize=7.5)
ax.axhline(y_prob_test.mean(), color='black', lw=1.5,
           linestyle=':', label=f'실제 테스트 평균={y_prob_test.mean():.3f}')
ax.set_ylabel('예측 P(가입)')
ax.set_title('시나리오별 평균 ± 표준편차')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

# (4) Diffusion vs MC — Normal 비교 (CDF)
ax = axes[1, 0]
for name, style in [('Diffusion — Normal','solid'), ('MC — Normal','dashed'),
                    ('Diffusion — Stress2','solid'), ('MC — Stress2','dashed')]:
    s = np.sort(probs[name])
    cdf = np.linspace(0, 1, len(s))
    ax.plot(s, cdf, color=COLORS[name], lw=2, linestyle=style, label=name)
ax.axvline(0.5, color='black', lw=1, linestyle='--')
ax.set_title('누적 분포 (CDF) 비교')
ax.set_xlabel('P(가입)')
ax.set_ylabel('누적 확률')
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3)

# (5) Diffusion 생성 데이터 duration 분포 vs 실제
ax = axes[1, 1]
ax.hist(X_train['duration'], bins=50, alpha=0.4, color='gray',
        density=True, label='실제 훈련 데이터')
ax.hist(df_diff_normal['duration'],  bins=50, alpha=0.5, color=COLORS['Diffusion — Normal'],
        density=True, label='Diffusion Normal')
ax.hist(df_diff_stress2['duration'], bins=50, alpha=0.5, color=COLORS['Diffusion — Stress2'],
        density=True, label='Diffusion Stress2')
ax.set_xlabel('duration (통화 시간)')
ax.set_title('Diffusion 생성 분포 vs 실제 — duration')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (6) 학습 Loss 곡선
ax = axes[1, 2]
ax.plot(range(1, EPOCHS+1), losses, color='steelblue', lw=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('DDPM 학습 곡선')
ax.grid(alpha=0.3)

plt.suptitle('스트레스 테스트 비교: Diffusion Model vs Monte Carlo', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('output/stress_test_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 피처별 스트레스 영향 분석 ────────────────────────────────
fig2, axes2 = plt.subplots(2, 4, figsize=(18, 9))
axes2 = axes2.flatten()

for i, col in enumerate(NUM_COLS):
    ax = axes2[i]
    ax.hist(X_train[col], bins=40, alpha=0.4, color='gray', density=True, label='실제')
    ax.hist(df_diff_stress2[col], bins=40, alpha=0.5,
            color=COLORS['Diffusion — Stress2'], density=True, label='Diff Stress2')
    ax.hist(df_mc_stress2[col],   bins=40, alpha=0.5,
            color=COLORS['MC — Stress2'], density=True, label='MC Stress2')
    ax.set_xlabel(col)
    ax.set_title(f'{col}')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

axes2[-1].set_visible(False)
plt.suptitle('극단 스트레스 시나리오의 피처 분포 (Diffusion vs MC)', fontsize=14)
plt.tight_layout()
plt.savefig('output/stress_feature_dist.png', dpi=150)
plt.close()

print("\n저장 완료:")
print("  output/stress_test_comparison.png")
print("  output/stress_feature_dist.png")

# ─── 최종 해석 요약 ───────────────────────────────────────────
print("\n" + "═"*65)
print("【해석 요약】")
print("─"*65)
d_n  = probs['Diffusion — Normal'].mean()
d_s2 = probs['Diffusion — Stress2'].mean()
m_n  = probs['MC — Normal'].mean()
m_s2 = probs['MC — Stress2'].mean()
print(f"  Diffusion 스트레스 증가폭: {d_n:.4f} → {d_s2:.4f}  (Δ={d_s2-d_n:+.4f})")
print(f"  MC        스트레스 증가폭: {m_n:.4f} → {m_s2:.4f}  (Δ={m_s2-m_n:+.4f})")
print()
print("  Diffusion: 데이터 매니폴드를 학습해 현실적인 극단 시나리오 생성")
print("             → 실제로 일어날 법한 스트레스 상황 반영")
print("  MC:        각 피처를 독립적으로 꼬리 분포에서 샘플링")
print("             → 피처 간 상관관계를 무시한 극단 조합 발생 가능")
print("═"*65)
