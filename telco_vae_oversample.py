"""
Telco Churn - VAE(Variational Autoencoder) 기반 오버샘플링
원본 데이터와 VAE 증강 데이터를 다각도로 비교·분석
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── 재현성 고정 ───────────────────────────────────────────────────
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
BETA       = 1.0   # KL 가중치

# ═══════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════════════════════
df = pd.read_csv("data/telco_churn_preprocessed.csv")
feature_names = [c for c in df.columns if c != "Churn"]

X = df[feature_names].values.astype(np.float32)
y = df["Churn"].values.astype(int)

n_majority = (y == 0).sum()   # 5174
n_minority = (y == 1).sum()   # 1869
n_generate = n_majority - n_minority   # 3305

print("=" * 55)
print(" VAE 기반 오버샘플링 - Telco Churn")
print("=" * 55)
print(f"\n[원본 데이터]")
print(f"  전체 샘플   : {len(y):,}개")
print(f"  Churn=0 (다수): {n_majority:,}개 ({n_majority/len(y)*100:.1f}%)")
print(f"  Churn=1 (소수): {n_minority:,}개 ({n_minority/len(y)*100:.1f}%)")
print(f"  불균형 비율  : 1 : {n_majority/n_minority:.2f}")
print(f"\n  생성 목표   : {n_generate:,}개 → 50:50 달성")

# ═══════════════════════════════════════════════════════════════════
# 2. VAE 정의
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
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var


def vae_loss(recon, x, mu, log_var, beta=BETA):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
    kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# ═══════════════════════════════════════════════════════════════════
# 3. 소수 클래스만으로 VAE 학습
# ═══════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_minority = X[y == 1]
X_majority  = X[y == 0]

tensor_min = torch.tensor(X_minority, dtype=torch.float32)
loader = DataLoader(TensorDataset(tensor_min), batch_size=BATCH_SIZE, shuffle=True)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"\n[VAE 학습] 디바이스: {device}")

total_losses, recon_losses, kl_losses = [], [], []
model.train()
for epoch in range(EPOCHS):
    ep_total = ep_recon = ep_kl = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        recon, mu, log_var = model(batch)
        loss, rl, kl = vae_loss(recon, batch, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ep_total += loss.item()
        ep_recon += rl.item()
        ep_kl    += kl.item()
    n = len(loader)
    total_losses.append(ep_total / n)
    recon_losses.append(ep_recon / n)
    kl_losses.append(ep_kl / n)
    if (epoch + 1) % 75 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
              f"Total={ep_total/n:.1f}  Recon={ep_recon/n:.1f}  KL={ep_kl/n:.1f}")

# ═══════════════════════════════════════════════════════════════════
# 4. 합성 샘플 생성
# ═══════════════════════════════════════════════════════════════════
model.eval()
with torch.no_grad():
    # 잠재 공간에서 직접 샘플링 (N(0,I))
    z_sample = torch.randn(n_generate, LATENT_DIM).to(device)
    X_synth  = model.decode(z_sample).cpu().numpy()

    # 실제 소수 클래스 잠재 벡터 (시각화용)
    mu_real, lv_real = model.encode(tensor_min.to(device))
    z_real = model.reparameterize(mu_real, lv_real).cpu().numpy()

# 오버샘플링 데이터셋 구성
X_aug = np.vstack([X, X_synth])
y_aug = np.concatenate([y, np.ones(n_generate, dtype=int)])

print(f"\n[생성 완료] 합성 샘플: {n_generate:,}개")
print(f"\n[오버샘플링 후]")
print(f"  Churn=0: {(y_aug==0).sum():,}개 (50.0%)")
print(f"  Churn=1: {(y_aug==1).sum():,}개 (50.0%)")

# ═══════════════════════════════════════════════════════════════════
# 5. 통계 비교 (원본 소수 vs 합성)
# ═══════════════════════════════════════════════════════════════════
df_real  = pd.DataFrame(X_minority, columns=feature_names)
df_synth = pd.DataFrame(X_synth,    columns=feature_names)

print("\n" + "=" * 55)
print(" 원본 소수 클래스 vs VAE 합성 샘플 통계 비교")
print("=" * 55)
print(f"{'Feature':<20} {'Real Mean':>10} {'Synth Mean':>10} "
      f"{'Real Std':>10} {'Synth Std':>10}")
print("-" * 60)
for col in feature_names:
    rm  = df_real[col].mean()
    sm  = df_synth[col].mean()
    rs  = df_real[col].std()
    ss  = df_synth[col].std()
    print(f"{col:<20} {rm:>10.4f} {sm:>10.4f} {rs:>10.4f} {ss:>10.4f}")

# ═══════════════════════════════════════════════════════════════════
# 6. 분류 성능 비교
# ═══════════════════════════════════════════════════════════════════
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

X_tr_aug = np.vstack([X_tr, X_synth])
y_tr_aug = np.concatenate([y_tr, np.ones(n_generate, dtype=int)])

def train_eval(X_train, y_train, label):
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight=None)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    return {
        "label":   label,
        "clf":     clf,
        "y_pred":  y_pred,
        "y_prob":  y_prob,
        "auc":     roc_auc_score(y_te, y_prob),
        "f1":      f1_score(y_te, y_pred),
        "cm":      confusion_matrix(y_te, y_pred),
        "report":  classification_report(y_te, y_pred,
                       target_names=["No Churn", "Churn"]),
    }

res_orig = train_eval(X_tr,     y_tr,     "원본 데이터")
res_vae  = train_eval(X_tr_aug, y_tr_aug, "VAE 오버샘플링")

print("\n" + "=" * 55)
print(" 분류 성능 비교 (RandomForest, test 20%)")
print("=" * 55)
for r in [res_orig, res_vae]:
    print(f"\n── {r['label']} ──")
    print(r["report"])
    print(f"  ROC-AUC : {r['auc']:.4f}")
    print(f"  F1(Churn): {r['f1']:.4f}")

print("\n" + "=" * 55)
print(f"  AUC  개선: {res_orig['auc']:.4f} → {res_vae['auc']:.4f}  "
      f"({res_vae['auc']-res_orig['auc']:+.4f})")
print(f"  F1   개선: {res_orig['f1']:.4f} → {res_vae['f1']:.4f}  "
      f"({res_vae['f1']-res_orig['f1']:+.4f})")
print("=" * 55)

# ═══════════════════════════════════════════════════════════════════
# 7. 시각화 (7-패널 대시보드)
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14))
fig.suptitle("VAE Oversampling — Telco Churn 원본 vs 증강 비교",
             fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.35)

colors = {"real_maj": "#4878CF", "real_min": "#DD8452",
          "synth": "#59A14F",    "orig": "#4878CF", "vae": "#DD8452"}

# ── (A) VAE 학습 손실 ─────────────────────────────────────────────
ax_loss = fig.add_subplot(gs[0, 0])
ax_loss.plot(total_losses, label="Total",       color="#4878CF")
ax_loss.plot(recon_losses, label="Recon (MSE)", color="#DD8452", linestyle="--")
ax_loss.plot(kl_losses,    label="KL",          color="#59A14F", linestyle=":")
ax_loss.set_title("(A) VAE Training Loss", fontweight="bold")
ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss / N")
ax_loss.legend(fontsize=8); ax_loss.grid(True, alpha=0.3)

# ── (B) 클래스 분포 ───────────────────────────────────────────────
ax_dist = fig.add_subplot(gs[0, 1])
categories = ["No Churn (0)", "Churn (1)"]
orig_c = [n_majority, n_minority]
aug_c  = [n_majority, n_minority + n_generate]
x = np.arange(2); w = 0.35
ax_dist.bar(x - w/2, orig_c, w, label="Original", color=colors["orig"], alpha=0.85)
ax_dist.bar(x + w/2, aug_c,  w, label="Augmented (VAE)", color=colors["vae"], alpha=0.85)
for xi, v in zip(x - w/2, orig_c): ax_dist.text(xi, v+50, str(v), ha="center", fontsize=8)
for xi, v in zip(x + w/2, aug_c):  ax_dist.text(xi, v+50, str(v), ha="center", fontsize=8)
ax_dist.set_xticks(x); ax_dist.set_xticklabels(categories)
ax_dist.set_title("(B) Class Distribution", fontweight="bold")
ax_dist.legend(fontsize=8); ax_dist.grid(True, alpha=0.3, axis="y")

# ── (C) 잠재 공간 (PCA 2D) ───────────────────────────────────────
ax_latent = fig.add_subplot(gs[0, 2])
pca = PCA(n_components=2, random_state=SEED)
z_all = np.vstack([z_real, z_sample.cpu().numpy()])
z_2d  = pca.fit_transform(z_all)
n_r   = len(z_real)
ax_latent.scatter(z_2d[:n_r, 0], z_2d[:n_r, 1],
                  alpha=0.5, s=10, label="Real minority", color=colors["real_min"])
ax_latent.scatter(z_2d[n_r:, 0], z_2d[n_r:, 1],
                  alpha=0.2, s=5,  label="Synthetic",      color=colors["synth"])
ax_latent.set_title("(C) Latent Space (PCA 2D)", fontweight="bold")
ax_latent.set_xlabel("PC1"); ax_latent.set_ylabel("PC2")
ax_latent.legend(fontsize=8); ax_latent.grid(True, alpha=0.3)

# ── (D) t-SNE (원본 전체) ─────────────────────────────────────────
ax_tsne = fig.add_subplot(gs[0, 3])
idx_maj = np.random.choice(np.where(y==0)[0], 300, replace=False)
idx_min = np.where(y==1)[0]
idx_syn = np.random.choice(n_generate, 300, replace=False)
X_vis = np.vstack([X[idx_maj], X[idx_min], X_synth[idx_syn]])
l_vis = np.array([0]*300 + [1]*len(idx_min) + [2]*300)
tsne  = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=500)
X_2d  = tsne.fit_transform(X_vis)
for lbl, name, col, sz in [(0,"No Churn (real)",colors["real_maj"],8),
                             (1,"Churn (real)",  colors["real_min"],10),
                             (2,"Churn (synth)", colors["synth"],   6)]:
    m = l_vis == lbl
    ax_tsne.scatter(X_2d[m,0], X_2d[m,1], alpha=0.5, s=sz, label=name, color=col)
ax_tsne.set_title("(D) t-SNE (Feature Space)", fontweight="bold")
ax_tsne.legend(fontsize=7); ax_tsne.grid(True, alpha=0.3)

# ── (E) 주요 피처 분포 비교 (연속형) ────────────────────────────
cont_feats = ["tenure", "MonthlyCharges", "TotalCharges"]
for i, feat in enumerate(cont_feats):
    ax = fig.add_subplot(gs[1, i])
    col_idx = feature_names.index(feat)
    ax.hist(X_minority[:, col_idx], bins=25, alpha=0.6, density=True,
            label="Real minority", color=colors["real_min"])
    ax.hist(X_synth[:, col_idx],    bins=25, alpha=0.5, density=True,
            label="Synthetic",      color=colors["synth"])
    ax.set_title(f"(E{i+1}) {feat}", fontweight="bold")
    ax.set_ylabel("Density"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── (F) 이진 피처 평균 비교 (막대) ──────────────────────────────
ax_bin = fig.add_subplot(gs[1, 3])
bin_feats = ["SeniorCitizen","Partner","Dependents","PhoneService",
             "PaperlessBilling","OnlineSecurity","TechSupport"]
real_means  = [X_minority[:, feature_names.index(f)].mean() for f in bin_feats]
synth_means = [X_synth[:,    feature_names.index(f)].mean() for f in bin_feats]
x2 = np.arange(len(bin_feats)); w2 = 0.35
ax_bin.barh(x2 - w2/2, real_means,  w2, label="Real",      color=colors["real_min"], alpha=0.85)
ax_bin.barh(x2 + w2/2, synth_means, w2, label="Synthetic", color=colors["synth"],    alpha=0.85)
ax_bin.set_yticks(x2); ax_bin.set_yticklabels(bin_feats, fontsize=8)
ax_bin.set_title("(F) Binary Feature Means", fontweight="bold")
ax_bin.set_xlabel("Mean Value"); ax_bin.legend(fontsize=8); ax_bin.grid(True, alpha=0.3, axis="x")

# ── (G) 혼동 행렬 비교 ───────────────────────────────────────────
for i, res in enumerate([res_orig, res_vae]):
    ax_cm = fig.add_subplot(gs[2, i])
    cm = res["cm"]
    im = ax_cm.imshow(cm, cmap="Blues")
    for r in range(2):
        for c in range(2):
            ax_cm.text(c, r, str(cm[r, c]), ha="center", va="center",
                       fontsize=13, fontweight="bold",
                       color="white" if cm[r, c] > cm.max()/2 else "black")
    ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
    ax_cm.set_xticklabels(["No Churn","Churn"]); ax_cm.set_yticklabels(["No Churn","Churn"])
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
    ax_cm.set_title(f"(G{i+1}) Confusion Matrix\n{res['label']}\n"
                    f"AUC={res['auc']:.4f}  F1={res['f1']:.4f}", fontweight="bold")

# ── (H) ROC 곡선 비교 ────────────────────────────────────────────
from sklearn.metrics import roc_curve
ax_roc = fig.add_subplot(gs[2, 2])
for res, col in [(res_orig, colors["orig"]), (res_vae, colors["vae"])]:
    fpr, tpr, _ = roc_curve(y_te, res["y_prob"])
    ax_roc.plot(fpr, tpr, color=col,
                label=f"{res['label']} (AUC={res['auc']:.4f})", linewidth=2)
ax_roc.plot([0,1],[0,1], "k--", linewidth=1)
ax_roc.set_title("(H) ROC Curve", fontweight="bold")
ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
ax_roc.legend(fontsize=9); ax_roc.grid(True, alpha=0.3)

# ── (I) 성능 지표 요약 텍스트 ────────────────────────────────────
ax_sum = fig.add_subplot(gs[2, 3])
ax_sum.axis("off")

from sklearn.metrics import precision_score, recall_score
metrics = {
    "AUC":       (res_orig["auc"],                   res_vae["auc"]),
    "F1(Churn)": (res_orig["f1"],                    res_vae["f1"]),
    "Precision": (precision_score(y_te, res_orig["y_pred"]),
                  precision_score(y_te, res_vae["y_pred"])),
    "Recall":    (recall_score(y_te, res_orig["y_pred"]),
                  recall_score(y_te, res_vae["y_pred"])),
}

lines = ["(I) 성능 요약 비교\n",
         f"{'지표':<12}{'원본':>8}{'VAE':>8}{'개선':>8}",
         "─" * 38]
for k, (o, v) in metrics.items():
    diff = v - o
    arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "─")
    lines.append(f"{k:<12}{o:>8.4f}{v:>8.4f}  {arrow}{abs(diff):.4f}")

lines += ["─" * 38,
          f"\n원본  불균형: {n_majority}:{n_minority}",
          f"증강 후 균형: {n_majority}:{n_minority+n_generate}",
          f"합성 샘플 수: {n_generate:,}개"]

ax_sum.text(0.05, 0.95, "\n".join(lines),
            transform=ax_sum.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="aliceblue", alpha=0.8))

plt.savefig("output/telco_vae_oversample.png", dpi=150, bbox_inches="tight")
print("\n[시각화 저장] output/telco_vae_oversample.png")
