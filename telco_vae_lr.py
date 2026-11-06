"""
Telco Churn - VAE 오버샘플링 + Logistic Regression 평가
원본 vs VAE 증강 데이터를 로지스틱 회귀로 비교
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 하이퍼파라미터 ────────────────────────────────────────────────
INPUT_DIM  = 19
LATENT_DIM = 10
HIDDEN_DIM = 64
EPOCHS     = 300
BATCH_SIZE = 64
LR         = 1e-3
BETA       = 1.0

# ═══════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════════════════════
df = pd.read_csv("data/telco_churn_preprocessed.csv")
feature_names = [c for c in df.columns if c != "Churn"]

X = df[feature_names].values.astype(np.float32)
y = df["Churn"].values.astype(int)

n_majority = (y == 0).sum()
n_minority = (y == 1).sum()
n_generate = n_majority - n_minority

print("=" * 55)
print(" VAE + Logistic Regression 평가 - Telco Churn")
print("=" * 55)
print(f"\n[원본 데이터]")
print(f"  Churn=0 (다수): {n_majority:,}개 ({n_majority/len(y)*100:.1f}%)")
print(f"  Churn=1 (소수): {n_minority:,}개 ({n_minority/len(y)*100:.1f}%)")
print(f"  불균형 비율   : 1 : {n_majority/n_minority:.2f}")
print(f"  생성 목표     : {n_generate:,}개")

# ═══════════════════════════════════════════════════════════════════
# 2. VAE 정의 & 학습
# ═══════════════════════════════════════════════════════════════════
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu      = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_log_var = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, INPUT_DIM),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def vae_loss(recon, x, mu, log_var):
    rl = nn.functional.mse_loss(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return rl + BETA * kl, rl, kl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_minority = X[y == 1]

tensor_min = torch.tensor(X_minority, dtype=torch.float32)
loader = DataLoader(TensorDataset(tensor_min), batch_size=BATCH_SIZE, shuffle=True)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"\n[VAE 학습] 디바이스: {device}")
losses = []
model.train()
for epoch in range(EPOCHS):
    ep = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        recon, mu, lv = model(batch)
        loss, _, _ = vae_loss(recon, batch, mu, lv)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        ep += loss.item()
    losses.append(ep / len(loader))
    if (epoch + 1) % 75 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  Loss={losses[-1]:.1f}")

# ═══════════════════════════════════════════════════════════════════
# 3. 합성 샘플 생성
# ═══════════════════════════════════════════════════════════════════
model.eval()
with torch.no_grad():
    z_sample = torch.randn(n_generate, LATENT_DIM).to(device)
    X_synth  = model.decode(z_sample).cpu().numpy()

X_aug = np.vstack([X, X_synth])
y_aug = np.concatenate([y, np.ones(n_generate, dtype=int)])

print(f"\n[생성 완료] 합성 샘플: {n_generate:,}개 → 총 {len(y_aug):,}개")

# ═══════════════════════════════════════════════════════════════════
# 4. Logistic Regression 평가 (4가지 설정 비교)
# ═══════════════════════════════════════════════════════════════════
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

X_tr_aug = np.vstack([X_tr, X_synth])
y_tr_aug = np.concatenate([y_tr, np.ones(n_generate, dtype=int)])

configs = [
    ("원본 (기본)",          X_tr,     y_tr,     None),
    ("원본 + class_weight",  X_tr,     y_tr,     "balanced"),
    ("VAE 오버샘플링",       X_tr_aug, y_tr_aug, None),
    ("VAE + class_weight",  X_tr_aug, y_tr_aug, "balanced"),
]

results = []
print("\n" + "=" * 55)
print(" Logistic Regression 성능 비교")
print("=" * 55)

for label, Xtr, ytr, cw in configs:
    clf = LogisticRegression(max_iter=1000, random_state=SEED,
                             class_weight=cw, C=1.0)
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    # 5-Fold CV AUC
    cv_clf = LogisticRegression(max_iter=1000, random_state=SEED,
                                class_weight=cw, C=1.0)
    cv_auc = cross_val_score(cv_clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                             scoring="roc_auc").mean()

    res = {
        "label":    label,
        "clf":      clf,
        "y_pred":   y_pred,
        "y_prob":   y_prob,
        "auc":      roc_auc_score(y_te, y_prob),
        "cv_auc":   cv_auc,
        "f1":       f1_score(y_te, y_pred),
        "precision":precision_score(y_te, y_pred),
        "recall":   recall_score(y_te, y_pred),
        "cm":       confusion_matrix(y_te, y_pred),
        "report":   classification_report(y_te, y_pred,
                        target_names=["No Churn","Churn"]),
    }
    results.append(res)
    print(f"\n── {label} ──")
    print(res["report"])
    print(f"  Test AUC : {res['auc']:.4f}   5-Fold CV AUC: {res['cv_auc']:.4f}")
    print(f"  F1       : {res['f1']:.4f}   Precision: {res['precision']:.4f}   Recall: {res['recall']:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 5. 계수(Coefficient) 비교: 원본 vs VAE 증강
# ═══════════════════════════════════════════════════════════════════
coef_orig = results[0]["clf"].coef_[0]
coef_vae  = results[2]["clf"].coef_[0]

print("\n" + "=" * 55)
print(" LR 계수 비교 (원본 vs VAE 증강) - 절댓값 내림차순")
print("=" * 55)
print(f"{'Feature':<20} {'Orig Coef':>10} {'VAE Coef':>10} {'Diff':>10}")
print("-" * 52)
order = np.argsort(np.abs(coef_vae))[::-1]
for i in order:
    diff = coef_vae[i] - coef_orig[i]
    print(f"{feature_names[i]:<20} {coef_orig[i]:>10.4f} {coef_vae[i]:>10.4f} {diff:>+10.4f}")

# ═══════════════════════════════════════════════════════════════════
# 6. 시각화
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14))
fig.suptitle("VAE Oversampling + Logistic Regression — Telco Churn",
             fontsize=14, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

palette = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]

# ── (A) 학습 손실 ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(losses, color="#4878CF", linewidth=1.5)
ax.set_title("(A) VAE Training Loss", fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss / N")
ax.grid(True, alpha=0.3)

# ── (B) 4가지 설정 AUC 비교 ─────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
labels_short = ["원본", "원본+CW", "VAE", "VAE+CW"]
auc_vals  = [r["auc"]    for r in results]
cv_vals   = [r["cv_auc"] for r in results]
x = np.arange(4); w = 0.35
ax.bar(x - w/2, auc_vals, w, label="Test AUC",   color=palette, alpha=0.85)
ax.bar(x + w/2, cv_vals,  w, label="CV AUC",     color=palette, alpha=0.45, hatch="//")
for xi, v in enumerate(auc_vals): ax.text(xi - w/2, v+0.002, f"{v:.4f}", ha="center", fontsize=7.5, fontweight="bold")
for xi, v in enumerate(cv_vals):  ax.text(xi + w/2, v+0.002, f"{v:.4f}", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(labels_short, fontsize=9)
ax.set_ylim(0.75, 0.90)
ax.set_title("(B) AUC 비교 (4가지 설정)", fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

# ── (C) F1 / Precision / Recall 비교 ─────────────────────────────
ax = fig.add_subplot(gs[0, 2])
metrics_keys = ["f1", "precision", "recall"]
metric_labels = ["F1", "Precision", "Recall"]
x = np.arange(4); w = 0.22
for i, (k, ml) in enumerate(zip(metrics_keys, metric_labels)):
    vals = [r[k] for r in results]
    ax.bar(x + (i-1)*w, vals, w, label=ml, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(labels_short, fontsize=9)
ax.set_ylim(0, 0.85)
ax.set_title("(C) F1 / Precision / Recall", fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

# ── (D) ROC 곡선 (4가지) ─────────────────────────────────────────
ax = fig.add_subplot(gs[0, 3])
for res, col in zip(results, palette):
    fpr, tpr, _ = roc_curve(y_te, res["y_prob"])
    ax.plot(fpr, tpr, color=col, linewidth=2,
            label=f"{res['label']} ({res['auc']:.4f})")
ax.plot([0,1],[0,1], "k--", linewidth=1)
ax.set_title("(D) ROC Curve", fontweight="bold")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)

# ── (E1~E4) 혼동 행렬 4개 ────────────────────────────────────────
for i, res in enumerate(results):
    ax = fig.add_subplot(gs[1, i])
    cm = res["cm"]
    im = ax.imshow(cm, cmap="Blues")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if cm[r,c] > cm.max()/2 else "black")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No","Churn"]); ax.set_yticklabels(["No","Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(f"(E{i+1}) {res['label']}\n"
                 f"AUC={res['auc']:.4f}  F1={res['f1']:.4f}\n"
                 f"TP={tp}  FN={fn}", fontweight="bold", fontsize=9)

# ── (F) LR 계수 비교 (상위 10개) ─────────────────────────────────
ax = fig.add_subplot(gs[2, :2])
top_idx = np.argsort(np.abs(coef_vae))[::-1][:10]
feat_top = [feature_names[i] for i in top_idx]
y_pos = np.arange(len(feat_top))
w = 0.35
ax.barh(y_pos - w/2, coef_orig[top_idx], w, label="원본 LR 계수",   color="#4878CF", alpha=0.85)
ax.barh(y_pos + w/2, coef_vae[top_idx],  w, label="VAE LR 계수",   color="#D65F5F", alpha=0.85)
ax.set_yticks(y_pos); ax.set_yticklabels(feat_top, fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("(F) LR 계수 비교 (절댓값 상위 10개) — 원본 vs VAE 증강",
             fontweight="bold")
ax.set_xlabel("Coefficient"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="x")

# ── (G) 성능 요약 테이블 ─────────────────────────────────────────
ax = fig.add_subplot(gs[2, 2:])
ax.axis("off")

header = f"{'설정':<18} {'AUC':>7} {'CV-AUC':>7} {'F1':>7} {'Prec':>7} {'Recall':>7}"
lines  = ["(G) Logistic Regression 성능 요약\n",
          header, "─" * 56]
for r in results:
    lines.append(
        f"{r['label']:<18} {r['auc']:>7.4f} {r['cv_auc']:>7.4f} "
        f"{r['f1']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f}"
    )

# 최고 설정 강조
best = max(results, key=lambda r: r["f1"])
lines += ["─" * 56,
          f"  최고 F1 설정: {best['label']}  (F1={best['f1']:.4f})",
          "",
          f"  원본 불균형 : {n_majority}:{n_minority}  →  불균형비 1:{n_majority/n_minority:.2f}",
          f"  VAE 증강 후 : {n_majority}:{n_minority+n_generate}  →  균형비 1:1",
          f"  합성 샘플 수: {n_generate:,}개",
          "",
          "  ※ LR은 RF보다 불균형에 민감 → VAE+class_weight 조합 권장"]

ax.text(0.02, 0.96, "\n".join(lines),
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="aliceblue", alpha=0.8))

plt.savefig("output/telco_vae_lr.png", dpi=150, bbox_inches="tight")
print("\n[시각화 저장] output/telco_vae_lr.png")
