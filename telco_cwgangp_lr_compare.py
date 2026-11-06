"""
Telco Churn - Conditional WGAN-GP 오버샘플링
① 원본 데이터        → Logistic Regression 학습·평가
② CWGAN-GP 증강 데이터 → Logistic Regression 학습·평가
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
# 하이퍼파라미터
# ═══════════════════════════════════════════════════════
INPUT_DIM   = 19
LATENT_DIM  = 32
HIDDEN_DIM  = 128
N_CLASSES   = 2
EPOCHS      = 300
BATCH       = 256
LR_G        = 1e-4
LR_D        = 1e-4
N_CRITIC    = 5       # Discriminator를 Generator보다 5배 더 학습
LAMBDA_GP   = 10      # Gradient Penalty 가중치

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

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

print("=" * 54)
print("  Conditional WGAN-GP + Logistic Regression 비교")
print("=" * 54)
print(f"\n[원본 클래스 분포]")
print(f"  Churn=0 (다수) : {n_maj:,}개  ({n_maj/len(y)*100:.1f}%)")
print(f"  Churn=1 (소수) : {n_min:,}개  ({n_min/len(y)*100:.1f}%)")
print(f"  불균형 비율    : 1 : {n_maj/n_min:.2f}")
print(f"  생성 목표      : {n_gen:,}개")

# ═══════════════════════════════════════════════════════
# 2. Conditional WGAN-GP 정의
# ═══════════════════════════════════════════════════════
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, N_CLASSES)
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + N_CLASSES, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM // 2, INPUT_DIM),
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, N_CLASSES)
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM + N_CLASSES, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, x, labels):
        c = self.label_emb(labels)
        inp = torch.cat([x, c], dim=1)
        return self.net(inp)


def gradient_penalty(D, real, fake, labels, device):
    """WGAN-GP Gradient Penalty"""
    bs = real.size(0)
    alpha = torch.rand(bs, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interp, labels)
    grad = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True)[0]
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# ═══════════════════════════════════════════════════════
# 3. CWGAN-GP 학습 (전체 데이터 사용, 조건부로 소수 클래스 생성)
# ═══════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습용: train 데이터 전체
t_X = torch.tensor(X_tr, dtype=torch.float32)
t_y = torch.tensor(y_tr, dtype=torch.long)
loader = DataLoader(TensorDataset(t_X, t_y), batch_size=BATCH, shuffle=True, drop_last=True)

G = Generator().to(device)
D = Discriminator().to(device)
opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=(0.0, 0.9))
opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=(0.0, 0.9))

print(f"\n[CWGAN-GP 학습] 디바이스: {device}")

g_losses, d_losses = [], []
G.train(); D.train()

for ep in range(EPOCHS):
    ep_d = ep_g = 0.0
    n_batch = 0

    for real_x, real_y in loader:
        real_x = real_x.to(device)
        real_y = real_y.to(device)
        bs = real_x.size(0)

        # ── Discriminator 학습 (N_CRITIC 회) ──
        for _ in range(N_CRITIC):
            z = torch.randn(bs, LATENT_DIM, device=device)
            fake_x = G(z, real_y).detach()

            d_real = D(real_x, real_y).mean()
            d_fake = D(fake_x, real_y).mean()
            gp     = gradient_penalty(D, real_x, fake_x, real_y, device)

            d_loss = d_fake - d_real + LAMBDA_GP * gp
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()
            ep_d += d_loss.item()

        # ── Generator 학습 ──
        z      = torch.randn(bs, LATENT_DIM, device=device)
        fake_x = G(z, real_y)
        g_loss = -D(fake_x, real_y).mean()
        opt_G.zero_grad(); g_loss.backward(); opt_G.step()
        ep_g += g_loss.item()
        n_batch += 1

    g_losses.append(ep_g / n_batch)
    d_losses.append(ep_d / (n_batch * N_CRITIC))

    if (ep + 1) % 100 == 0:
        print(f"  Epoch {ep+1:3d}/{EPOCHS}  "
              f"D_loss={d_losses[-1]:+.4f}  G_loss={g_losses[-1]:+.4f}")

# ═══════════════════════════════════════════════════════
# 4. 소수 클래스(Churn=1) 합성 샘플 생성
# ═══════════════════════════════════════════════════════
G.eval()
with torch.no_grad():
    z_s    = torch.randn(n_gen, LATENT_DIM, device=device)
    lbl_s  = torch.ones(n_gen, dtype=torch.long, device=device)  # Churn=1
    X_synth = G(z_s, lbl_s).cpu().numpy()

# 증강 학습셋
X_tr_aug = np.vstack([X_tr, X_synth])
y_tr_aug = np.concatenate([y_tr, np.ones(n_gen, dtype=int)])

print(f"\n[생성 완료] 합성 샘플: {n_gen:,}개")
print(f"[증강 train] Churn=0: {(y_tr_aug==0).sum():,}  "
      f"Churn=1: {(y_tr_aug==1).sum():,}")

# ═══════════════════════════════════════════════════════
# 5. Logistic Regression — 원본 vs CWGAN-GP 증강
# ═══════════════════════════════════════════════════════
lr_orig = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
lr_gan  = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)

lr_orig.fit(X_tr,     y_tr)
lr_gan.fit(X_tr_aug,  y_tr_aug)

def get_metrics(clf):
    yp = clf.predict(X_te)
    yb = clf.predict_proba(X_te)[:, 1]
    cm = confusion_matrix(y_te, yp)
    tn, fp, fn, tp = cm.ravel()
    return {
        "yp": yp, "yb": yb, "cm": cm,
        "auc": roc_auc_score(y_te, yb),
        "f1":  f1_score(y_te, yp),
        "prec": precision_score(y_te, yp),
        "rec":  recall_score(y_te, yp),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "report": classification_report(y_te, yp,
                      target_names=["No Churn","Churn"]),
    }

mo = get_metrics(lr_orig)
mg = get_metrics(lr_gan)

print("\n" + "=" * 54)
print("  [원본 → LR]")
print("=" * 54)
print(mo["report"])
print(f"  AUC={mo['auc']:.4f}  F1={mo['f1']:.4f}  "
      f"Prec={mo['prec']:.4f}  Recall={mo['rec']:.4f}")
print(f"  TP={mo['tp']}  FN={mo['fn']}  FP={mo['fp']}  TN={mo['tn']}")

print("\n" + "=" * 54)
print("  [CWGAN-GP 증강 → LR]")
print("=" * 54)
print(mg["report"])
print(f"  AUC={mg['auc']:.4f}  F1={mg['f1']:.4f}  "
      f"Prec={mg['prec']:.4f}  Recall={mg['rec']:.4f}")
print(f"  TP={mg['tp']}  FN={mg['fn']}  FP={mg['fp']}  TN={mg['tn']}")

print("\n" + "=" * 54)
print("  개선 비교 (원본 → CWGAN-GP)")
print("=" * 54)
for k, label in [("auc","AUC"),("f1","F1(Churn)"),("prec","Precision"),("rec","Recall")]:
    diff = mg[k] - mo[k]
    sym  = "▲" if diff > 0.001 else ("▼" if diff < -0.001 else "─")
    print(f"  {label:<12}: {mo[k]:.4f}  →  {mg[k]:.4f}  ({sym}{abs(diff):.4f})")
print(f"\n  이탈 적중(TP): {mo['tp']} → {mg['tp']}  ({mg['tp']-mo['tp']:+d}명)")
print(f"  이탈 놓침(FN): {mo['fn']} → {mg['fn']}  ({mg['fn']-mo['fn']:+d}명)")

# ═══════════════════════════════════════════════════════
# 6. 시각화 (8-패널)
# ═══════════════════════════════════════════════════════
C_ORIG = "#4878CF"
C_GAN  = "#E84393"

fig = plt.figure(figsize=(20, 14))
fig.suptitle("Conditional WGAN-GP Oversampling + Logistic Regression\n"
             "원본 데이터  vs  CWGAN-GP 증강 데이터",
             fontsize=13, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

# ── (A) CWGAN-GP 학습 손실 ──────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(g_losses, color=C_GAN,  linewidth=1.5, label="G loss")
ax.plot(d_losses, color=C_ORIG, linewidth=1.5, label="D loss", linestyle="--")
ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax.set_title("(A) CWGAN-GP Training Loss", fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── (B) 클래스 분포 ────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
cats = ["No Churn (0)", "Churn (1)"]
ov = [n_maj, n_min]
gv = [n_maj, n_min + n_gen]
x = np.arange(2); w = 0.35
b1 = ax.bar(x - w/2, ov, w, color=C_ORIG, alpha=0.85, label="원본")
b2 = ax.bar(x + w/2, gv, w, color=C_GAN,  alpha=0.85, label="CWGAN-GP 증강")
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+80,
            f"{int(bar.get_height()):,}", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
ax.set_title("(B) 클래스 분포 비교", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

# ── (C) ROC 곡선 ───────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
for m, col, lbl in [(mo, C_ORIG, "원본 LR"), (mg, C_GAN, "CWGAN-GP LR")]:
    fpr, tpr, _ = roc_curve(y_te, m["yb"])
    ax.plot(fpr, tpr, color=col, linewidth=2.2,
            label=f"{lbl}  (AUC={m['auc']:.4f})")
ax.plot([0,1],[0,1],"k--",linewidth=1)
ax.set_title("(C) ROC Curve", fontweight="bold")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── (D) 지표 막대 비교 ─────────────────────────────
ax = fig.add_subplot(gs[0, 3])
mnames = ["AUC", "F1", "Precision", "Recall"]
ov_ = [mo["auc"], mo["f1"], mo["prec"], mo["rec"]]
gv_ = [mg["auc"], mg["f1"], mg["prec"], mg["rec"]]
x = np.arange(4); w = 0.35
b1 = ax.bar(x-w/2, ov_, w, color=C_ORIG, alpha=0.85, label="원본 LR")
b2 = ax.bar(x+w/2, gv_, w, color=C_GAN,  alpha=0.85, label="CWGAN-GP LR")
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.006,
            f"{bar.get_height():.3f}", ha="center", fontsize=7.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(mnames)
ax.set_ylim(0, 1.05)
ax.set_title("(D) 성능 지표 비교", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

# ── (E1/E2) 혼동행렬 ───────────────────────────────
for ci, (m, lbl, col, cmap, tag) in enumerate([
        (mo, "원본 LR",        C_ORIG, "Blues", "E1"),
        (mg, "CWGAN-GP LR",   C_GAN,  "RdPu",  "E2")]):
    ax = fig.add_subplot(gs[1, ci*2 : ci*2+2])
    im = ax.imshow(m["cm"], cmap=cmap, vmin=0, vmax=m["cm"].max()*1.2)
    lbs = [["TN","FP"],["FN","TP"]]
    for r in range(2):
        for c_ in range(2):
            v = m["cm"][r, c_]
            ax.text(c_, r, f"{lbs[r][c_]}\n{v}",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if v > m["cm"].max()/2 else "black")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["예측: No Churn","예측: Churn"], fontsize=9)
    ax.set_yticklabels(["실제: No Churn","실제: Churn"], fontsize=9)
    ax.set_title(
        f"({tag}) 혼동행렬 — {lbl}\n"
        f"AUC={m['auc']:.4f}  F1={m['f1']:.4f}  Recall={m['rec']:.4f}\n"
        f"TP={m['tp']} (이탈 적중)  FN={m['fn']} (이탈 놓침)  FP={m['fp']}",
        fontweight="bold", fontsize=9, pad=8)

# ── (F) LR 계수 비교 ───────────────────────────────
ax = fig.add_subplot(gs[2, :3])
coef_o = lr_orig.coef_[0]
coef_g = lr_gan.coef_[0]
order  = np.argsort(np.abs(coef_g))[::-1]
y_pos  = np.arange(len(feat_cols))
w = 0.35
ax.barh(y_pos-w/2, coef_o[order], w, color=C_ORIG, alpha=0.85, label="원본 LR 계수")
ax.barh(y_pos+w/2, coef_g[order], w, color=C_GAN,  alpha=0.85, label="CWGAN-GP LR 계수")
ax.set_yticks(y_pos)
ax.set_yticklabels([feat_cols[i] for i in order], fontsize=9)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("(F) LR 계수 비교 — 원본 vs CWGAN-GP 증강  (|GAN 계수| 내림차순)",
             fontweight="bold")
ax.set_xlabel("Coefficient  (양수 = 이탈 ↑,  음수 = 이탈 ↓)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="x")

# ── (G) 결과 요약 ──────────────────────────────────
ax = fig.add_subplot(gs[2, 3])
ax.axis("off")

def arr(o, v):
    d = v - o
    s = "▲" if d > 0.001 else ("▼" if d < -0.001 else "─")
    return f"{s}{abs(d):.4f}"

lines = [
    "(G) 최종 비교 요약",
    "",
    f"{'지표':<11}{'원본 LR':>9}{'GAN LR':>9}{'개선':>9}",
    "─"*40,
    f"{'AUC':<11}{mo['auc']:>9.4f}{mg['auc']:>9.4f}  {arr(mo['auc'],mg['auc'])}",
    f"{'F1(Churn)':<11}{mo['f1']:>9.4f}{mg['f1']:>9.4f}  {arr(mo['f1'],mg['f1'])}",
    f"{'Precision':<11}{mo['prec']:>9.4f}{mg['prec']:>9.4f}  {arr(mo['prec'],mg['prec'])}",
    f"{'Recall':<11}{mo['rec']:>9.4f}{mg['rec']:>9.4f}  {arr(mo['rec'],mg['rec'])}",
    "─"*40,
    "",
    "[혼동행렬 핵심]",
    f"  이탈 적중(TP) : {mo['tp']} → {mg['tp']}  ({mg['tp']-mo['tp']:+d}명)",
    f"  이탈 놓침(FN) : {mo['fn']} → {mg['fn']}  ({mg['fn']-mo['fn']:+d}명)",
    f"  정상 오탐(FP) : {mo['fp']} → {mg['fp']}  ({mg['fp']-mo['fp']:+d}명)",
    "",
    "[데이터 규모]",
    f"  원본 train   : {len(y_tr):,}개",
    f"  GAN train    : {len(y_tr_aug):,}개 (+{n_gen:,})",
    f"  Test (공통)  : {len(y_te):,}개",
    "",
    "[모델] Conditional WGAN-GP",
    f"  G: z({LATENT_DIM}) + label → sample({INPUT_DIM})",
    f"  D: sample + label → Wasserstein score",
    f"  GP λ={LAMBDA_GP}, n_critic={N_CRITIC}",
]
ax.text(0.03, 0.97, "\n".join(lines),
        transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#fff0f5", alpha=0.9))

plt.savefig("output/telco_cwgangp_lr_compare.png", dpi=150, bbox_inches="tight")
print("\n[저장 완료] output/telco_cwgangp_lr_compare.png")
