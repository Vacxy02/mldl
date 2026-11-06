"""
Telco Churn
① 원본 데이터  → Logistic Regression 학습·평가
② VAE 증강 데이터 → Logistic Regression 학습·평가
두 결과를 나란히 비교
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score, roc_curve,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ═══════════════════════════════════════════════════════
# 1. 데이터 로드 & 분할
# ═══════════════════════════════════════════════════════
df = pd.read_csv("data/telco_churn_preprocessed.csv")
feat_cols = [c for c in df.columns if c != "Churn"]

X = df[feat_cols].values.astype(np.float32)
y = df["Churn"].values.astype(int)

n_maj = (y == 0).sum()   # 5174
n_min = (y == 1).sum()   # 1869
n_gen = n_maj - n_min    # 3305

# 원본 train/test 고정 (이후 모든 평가는 동일 test set 사용)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print("=" * 52)
print("  원본 데이터 vs VAE 증강 — LR 비교")
print("=" * 52)
print(f"\n[원본 클래스 분포]")
print(f"  Churn=0 : {n_maj:,}개  ({n_maj/len(y)*100:.1f}%)")
print(f"  Churn=1 : {n_min:,}개  ({n_min/len(y)*100:.1f}%)")
print(f"  불균형비 : 1 : {n_maj/n_min:.2f}")
print(f"\n[학습/테스트 분할]")
print(f"  Train : {len(y_tr):,}개  |  Test : {len(y_te):,}개")

# ═══════════════════════════════════════════════════════
# 2. VAE 정의
# ═══════════════════════════════════════════════════════
INPUT_DIM  = len(feat_cols)   # 19
LATENT_DIM = 10
HIDDEN_DIM = 64
EPOCHS     = 300
BATCH      = 64

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM), nn.BatchNorm1d(HIDDEN_DIM), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LeakyReLU(0.2),
        )
        self.mu_fc  = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.lv_fc  = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.dec = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.BatchNorm1d(HIDDEN_DIM), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, INPUT_DIM),
        )
    def encode(self, x):
        h = self.enc(x)
        return self.mu_fc(h), self.lv_fc(h)
    def reparameterize(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(lv)
    def decode(self, z):
        return self.dec(z)
    def forward(self, x):
        mu, lv = self.encode(x)
        return self.decode(self.reparameterize(mu, lv)), mu, lv

def elbo(recon, x, mu, lv):
    r = nn.functional.mse_loss(recon, x, reduction="sum")
    k = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
    return r + k

# ═══════════════════════════════════════════════════════
# 3. VAE 학습 (소수 클래스만)
# ═══════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_min  = X[y == 1]
t_min  = torch.tensor(X_min, dtype=torch.float32)
loader = DataLoader(TensorDataset(t_min), batch_size=BATCH, shuffle=True)

model = VAE().to(device)
opt   = optim.Adam(model.parameters(), lr=1e-3)

print(f"\n[VAE 학습] ({device})")
train_loss = []
model.train()
for ep in range(EPOCHS):
    ep_loss = 0.0
    for (b,) in loader:
        b = b.to(device)
        recon, mu, lv = model(b)
        loss = elbo(recon, b, mu, lv)
        opt.zero_grad(); loss.backward(); opt.step()
        ep_loss += loss.item()
    train_loss.append(ep_loss / len(loader))
    if (ep + 1) % 100 == 0:
        print(f"  Epoch {ep+1:3d}/{EPOCHS}  Loss={train_loss[-1]:.1f}")

# ═══════════════════════════════════════════════════════
# 4. 합성 샘플 생성
# ═══════════════════════════════════════════════════════
model.eval()
with torch.no_grad():
    z   = torch.randn(n_gen, LATENT_DIM).to(device)
    X_s = model.decode(z).cpu().numpy()   # 합성 샘플

# 증강 학습셋 (train만 증강, test는 그대로)
X_tr_aug = np.vstack([X_tr, X_s])
y_tr_aug = np.concatenate([y_tr, np.ones(n_gen, dtype=int)])

print(f"\n[합성 샘플] {n_gen:,}개 생성")
print(f"[증강 train] Churn=0: {(y_tr_aug==0).sum():,}  Churn=1: {(y_tr_aug==1).sum():,}")

# ═══════════════════════════════════════════════════════
# 5. Logistic Regression — 원본 vs VAE
# ═══════════════════════════════════════════════════════
lr_orig = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
lr_vae  = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)

lr_orig.fit(X_tr,     y_tr)
lr_vae.fit(X_tr_aug,  y_tr_aug)

def evaluate(clf, label):
    yp = clf.predict(X_te)
    yb = clf.predict_proba(X_te)[:, 1]
    print(f"\n{'='*52}")
    print(f"  [{label}]")
    print(f"{'='*52}")
    print(classification_report(y_te, yp, target_names=["No Churn", "Churn"]))
    tn, fp, fn, tp = confusion_matrix(y_te, yp).ravel()
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  AUC       : {roc_auc_score(y_te, yb):.4f}")
    print(f"  F1(Churn) : {f1_score(y_te, yp):.4f}")
    print(f"  Precision : {precision_score(y_te, yp):.4f}")
    print(f"  Recall    : {recall_score(y_te, yp):.4f}")
    return yp, yb

yp_o, yb_o = evaluate(lr_orig, "원본 → LR")
yp_v, yb_v = evaluate(lr_vae,  "VAE 증강 → LR")

# 개선 요약
print(f"\n{'='*52}")
print("  개선 비교 (원본 → VAE 증강)")
print(f"{'='*52}")
print(f"  AUC       : {roc_auc_score(y_te,yb_o):.4f}  →  {roc_auc_score(y_te,yb_v):.4f}"
      f"  ({roc_auc_score(y_te,yb_v)-roc_auc_score(y_te,yb_o):+.4f})")
print(f"  F1(Churn) : {f1_score(y_te,yp_o):.4f}  →  {f1_score(y_te,yp_v):.4f}"
      f"  ({f1_score(y_te,yp_v)-f1_score(y_te,yp_o):+.4f})")
print(f"  Precision : {precision_score(y_te,yp_o):.4f}  →  {precision_score(y_te,yp_v):.4f}"
      f"  ({precision_score(y_te,yp_v)-precision_score(y_te,yp_o):+.4f})")
print(f"  Recall    : {recall_score(y_te,yp_o):.4f}  →  {recall_score(y_te,yp_v):.4f}"
      f"  ({recall_score(y_te,yp_v)-recall_score(y_te,yp_o):+.4f})")

# ═══════════════════════════════════════════════════════
# 6. 시각화
# ═══════════════════════════════════════════════════════
C_ORIG = "#4878CF"
C_VAE  = "#D65F5F"

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Logistic Regression 비교:  원본 데이터  vs  VAE 증강 데이터",
             fontsize=14, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

# ── (A) VAE 학습 손실 ────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(train_loss, color=C_ORIG, linewidth=1.5)
ax.set_title("(A) VAE Training Loss", fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("ELBO Loss / N")
ax.grid(True, alpha=0.3)

# ── (B) 클래스 분포 ─────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
cats  = ["No Churn (0)", "Churn (1)"]
orig_ = [n_maj, n_min]
aug_  = [n_maj, n_min + n_gen]
x = np.arange(2); w = 0.35
b1 = ax.bar(x - w/2, orig_, w, color=C_ORIG, alpha=0.85, label="원본")
b2 = ax.bar(x + w/2, aug_,  w, color=C_VAE,  alpha=0.85, label="VAE 증강")
for bar in b1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+60,
                        str(int(bar.get_height())), ha="center", fontsize=8, fontweight="bold")
for bar in b2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+60,
                        str(int(bar.get_height())), ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
ax.set_title("(B) 클래스 분포 비교", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

# ── (C) ROC 곡선 ─────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
for yb, col, lbl in [(yb_o, C_ORIG, "원본 LR"), (yb_v, C_VAE, "VAE LR")]:
    fpr, tpr, _ = roc_curve(y_te, yb)
    auc = roc_auc_score(y_te, yb)
    ax.plot(fpr, tpr, color=col, linewidth=2.2, label=f"{lbl}  (AUC={auc:.4f})")
ax.plot([0,1],[0,1], "k--", linewidth=1)
ax.set_title("(C) ROC Curve", fontweight="bold")
ax.set_xlabel("FPR (False Positive Rate)")
ax.set_ylabel("TPR (True Positive Rate)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── (D) 지표 막대 비교 ──────────────────────────────
ax = fig.add_subplot(gs[0, 3])
metric_names = ["AUC", "F1", "Precision", "Recall"]
orig_vals = [
    roc_auc_score(y_te, yb_o),
    f1_score(y_te, yp_o),
    precision_score(y_te, yp_o),
    recall_score(y_te, yp_o),
]
vae_vals = [
    roc_auc_score(y_te, yb_v),
    f1_score(y_te, yp_v),
    precision_score(y_te, yp_v),
    recall_score(y_te, yp_v),
]
x = np.arange(4); w = 0.35
b1 = ax.bar(x - w/2, orig_vals, w, color=C_ORIG, alpha=0.85, label="원본 LR")
b2 = ax.bar(x + w/2, vae_vals,  w, color=C_VAE,  alpha=0.85, label="VAE LR")
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{bar.get_height():.3f}", ha="center", fontsize=7.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(metric_names, fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_title("(D) 성능 지표 비교", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

# ── (E1) 원본 혼동행렬 ──────────────────────────────
for col_idx, (yp, lbl, col) in enumerate([
        (yp_o, "원본 LR", C_ORIG), (yp_v, "VAE LR", C_VAE)]):
    ax = fig.add_subplot(gs[1, col_idx*2 : col_idx*2+2])
    cm = confusion_matrix(y_te, yp)
    tn, fp, fn, tp = cm.ravel()

    # 배경색: 원본=파랑계열, VAE=빨강계열
    cmap = "Blues" if col == C_ORIG else "Reds"
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max()*1.2)

    labels_cm = [["TN", "FP"], ["FN", "TP"]]
    for r in range(2):
        for c in range(2):
            v = cm[r, c]
            ax.text(c, r, f"{labels_cm[r][c]}\n{v}",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if v > cm.max()/2 else "black")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["예측: No Churn", "예측: Churn"], fontsize=9)
    ax.set_yticklabels(["실제: No Churn", "실제: Churn"], fontsize=9)
    auc = roc_auc_score(y_te, yb_o if col==C_ORIG else yb_v)
    f1  = f1_score(y_te, yp)
    rc  = recall_score(y_te, yp)
    ax.set_title(
        f"({'E1' if col==C_ORIG else 'E2'}) 혼동행렬 — {lbl}\n"
        f"AUC={auc:.4f}  F1={f1:.4f}  Recall={rc:.4f}\n"
        f"TP={tp} (이탈 적중)  FN={fn} (이탈 놓침)  FP={fp}  TN={tn}",
        fontweight="bold", fontsize=10, pad=8)

# ── (F) LR 계수 비교 ────────────────────────────────
ax = fig.add_subplot(gs[2, :3])
coef_o = lr_orig.coef_[0]
coef_v = lr_vae.coef_[0]
order  = np.argsort(np.abs(coef_v))[::-1]
y_pos  = np.arange(len(feat_cols))
w = 0.35
ax.barh(y_pos - w/2, coef_o[order], w, color=C_ORIG, alpha=0.85, label="원본 LR 계수")
ax.barh(y_pos + w/2, coef_v[order], w, color=C_VAE,  alpha=0.85, label="VAE LR 계수")
ax.set_yticks(y_pos)
ax.set_yticklabels([feat_cols[i] for i in order], fontsize=9)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("(F) LR 회귀 계수 비교 (|VAE 계수| 내림차순)",
             fontweight="bold")
ax.set_xlabel("Coefficient  (양수 = 이탈 ↑, 음수 = 이탈 ↓)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="x")

# ── (G) 결과 요약 텍스트 ─────────────────────────────
ax = fig.add_subplot(gs[2, 3])
ax.axis("off")

def arrow(o, v):
    d = v - o
    sym = "▲" if d > 0.001 else ("▼" if d < -0.001 else "─")
    return f"{sym}{abs(d):.4f}"

cm_o = confusion_matrix(y_te, yp_o).ravel()   # tn fp fn tp
cm_v = confusion_matrix(y_te, yp_v).ravel()

lines = [
    "(G) 최종 비교 요약",
    "",
    f"{'지표':<11}{'원본 LR':>9}{'VAE LR':>9}{'개선':>9}",
    "─" * 40,
    f"{'AUC':<11}{roc_auc_score(y_te,yb_o):>9.4f}{roc_auc_score(y_te,yb_v):>9.4f}"
    f"  {arrow(roc_auc_score(y_te,yb_o), roc_auc_score(y_te,yb_v))}",
    f"{'F1(Churn)':<11}{f1_score(y_te,yp_o):>9.4f}{f1_score(y_te,yp_v):>9.4f}"
    f"  {arrow(f1_score(y_te,yp_o), f1_score(y_te,yp_v))}",
    f"{'Precision':<11}{precision_score(y_te,yp_o):>9.4f}{precision_score(y_te,yp_v):>9.4f}"
    f"  {arrow(precision_score(y_te,yp_o), precision_score(y_te,yp_v))}",
    f"{'Recall':<11}{recall_score(y_te,yp_o):>9.4f}{recall_score(y_te,yp_v):>9.4f}"
    f"  {arrow(recall_score(y_te,yp_o), recall_score(y_te,yp_v))}",
    "─" * 40,
    "",
    f"[혼동행렬 핵심]",
    f"  이탈 적중(TP) : {int(cm_o[3])}  →  {int(cm_v[3])}",
    f"  이탈 놓침(FN) : {int(cm_o[2])}  →  {int(cm_v[2])}",
    f"  정상 오탐(FP) : {int(cm_o[1])}  →  {int(cm_v[1])}",
    "",
    f"[데이터 규모]",
    f"  원본 train  : {len(y_tr):,}개",
    f"  VAE train   : {len(y_tr_aug):,}개 (+{n_gen:,})",
    f"  Test (공통) : {len(y_te):,}개",
]
ax.text(0.03, 0.97, "\n".join(lines),
        transform=ax.transAxes, fontsize=8.5, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f4ff", alpha=0.9))

plt.savefig("output/telco_vae_lr_compare.png", dpi=150, bbox_inches="tight")
print("\n[저장 완료] output/telco_vae_lr_compare.png")
