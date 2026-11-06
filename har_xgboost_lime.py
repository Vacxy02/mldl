"""
================================================================================
UCI HAR PCA 데이터셋을 활용한 XGBoost 분류 및 LIME(Local Interpretable Model-agnostic Explanations) 해석
================================================================================
- 데이터 경로:
  - 특성 데이터: /Users/2donghak/Desktop/부캠/머신러닝/mldl/data/har_features_pca_90.csv (65개 PCA 주성분 피처)
  - 타깃 데이터: /Users/2donghak/Desktop/부캠/머신러닝/mldl/data/har_targets.csv (6가지 인간 행동 레이블: 1~6)
- 행동 클래스 (Activity Mapping):
  1: WALKING (걷기)
  2: WALKING_UPSTAIRS (계단 오르기)
  3: WALKING_DOWNSTAIRS (계단 내려가기)
  4: SITTING (앉아있기)
  5: STANDING (서있기)
  6: LAYING (누워있기)
- 주요 실행 흐름:
  1. 데이터 로드 및 0-based 레이블 인코딩
  2. 계층화 분할 (Train/Test Split: 80% / 20%)
  3. XGBoost(XGBClassifier) 모델 학습 및 테스트셋 성능 평가
  4. 테스트 데이터셋 중 임의의(Arbitrary) 샘플 선정 및 예측
  5. LIME Tabular Explainer를 통한 로컬 설명 생성 (주요 기여 피처 도출)
  6. 결과 해석 리포트 출력 및 시각화 차트(PNG/HTML) 저장
================================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import lime
import lime.lime_tabular

# 한글 폰트 설정 (macOS: AppleGothic, Windows: Malgun Gothic)
if sys.platform == "darwin":
    plt.rc("font", family="AppleGothic")
elif sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False

# 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "har_lime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = DATA_DIR / "har_features_pca_90.csv"
TARGETS_PATH = DATA_DIR / "har_targets.csv"

# 행동 클래스 정의
ACTIVITY_NAMES = [
    "WALKING",             # 0 (원래 라벨 1)
    "WALKING_UPSTAIRS",    # 1 (원래 라벨 2)
    "WALKING_DOWNSTAIRS",  # 2 (원래 라벨 3)
    "SITTING",             # 3 (원래 라벨 4)
    "STANDING",            # 4 (원래 라벨 5)
    "LAYING"               # 5 (원래 라벨 6)
]


def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    데이터 로드, 타깃 라벨(0~5) 변환 및 Train/Test 데이터 분할
    """
    print("=" * 70)
    print(" [1] 데이터셋 로드 및 전처리")
    print("=" * 70)
    
    if not FEATURES_PATH.exists() or not TARGETS_PATH.exists():
        raise FileNotFoundError(
            f"데이터 파일이 누락되었습니다.\n- {FEATURES_PATH}\n- {TARGETS_PATH}"
        )

    X = pd.read_csv(FEATURES_PATH)
    y_raw = pd.read_csv(TARGETS_PATH)["Activity"]

    print(f"  - PCA 특성 데이터 크기: {X.shape} (행: {X.shape[0]}, 주성분 수: {X.shape[1]})")
    print(f"  - 타깃 데이터 크기: {y_raw.shape}")

    # XGBoost는 0부터 시작하는 클래스 인덱스를 요구하므로 LabelEncoder 적용 (1~6 -> 0~5)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # 데이터셋 계층화 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"  - 학습셋(Train): {X_train.shape[0]}개, 평가셋(Test): {X_test.shape[0]}개")
    return X_train, X_test, y_train, y_test, label_encoder


def train_xgboost(X_train, y_train, random_state=42):
    """
    XGBoost 다중 분류기 학습
    """
    print("\n" + "=" * 70)
    print(" [2] XGBoost 분류 모델 학습")
    print("=" * 70)

    model = XGBClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("  >> XGBoost 모델 학습 완료")
    return model


def evaluate_model(model, X_test, y_test):
    """
    모델 평가 (정확도, 분류 리포트)
    """
    print("\n" + "=" * 70)
    print(" [3] 테스트 데이터셋 모델 평가")
    print("=" * 70)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  - 테스트 데이터 정확도(Accuracy): {accuracy * 100:.2f}%\n")

    report = classification_report(
        y_test, y_pred, target_names=ACTIVITY_NAMES, digits=4
    )
    print("  [분류 성능 상세 리포트]")
    print(report)


def explain_sample_with_lime(model, X_train, X_test, y_test, sample_idx=None, random_state=42):
    """
    임의의 테스트 샘플을 선정하고 LIME을 사용하여 예측 결과를 설명 및 시각화
    """
    print("=" * 70)
    print(" [4] LIME(Local Interpretable Model-agnostic Explanations) 분석")
    print("=" * 70)

    # 샘플 인덱스 선택 (지정되지 않은 경우 재현성을 위해 random_state 기반 임의 선정)
    if sample_idx is None:
        rng = np.random.RandomState(random_state)
        sample_idx = rng.randint(0, len(X_test))

    # 선정된 샘플 추출 (1D numpy array 형태)
    sample_series = X_test.iloc[sample_idx]
    sample_values = sample_series.to_numpy()
    true_class_idx = y_test[sample_idx]
    true_activity = ACTIVITY_NAMES[true_class_idx]

    # 모델 예측 및 클래스별 확률 계산
    prob_distribution = model.predict_proba(sample_values.reshape(1, -1))[0]
    pred_class_idx = int(np.argmax(prob_distribution))
    pred_activity = ACTIVITY_NAMES[pred_class_idx]
    pred_confidence = prob_distribution[pred_class_idx]

    print(f"  - 선택된 테스트 샘플 인덱스: {sample_idx}")
    print(f"  - 실제 정답(Ground Truth) : {true_activity} (라벨: {true_class_idx + 1})")
    print(f"  - 모델 예측(Prediction)   : {pred_activity} (라벨: {pred_class_idx + 1})")
    print(f"  - 예측 확신도(Confidence) : {pred_confidence * 100:.2f}%")
    print("\n  [전체 클래스별 예측 확률]")
    for i, act in enumerate(ACTIVITY_NAMES):
        bar = "■" * int(prob_distribution[i] * 30)
        print(f"    * {act:20s}: {prob_distribution[i]*100:6.2f}% | {bar}")

    # LIME Tabular Explainer 초기화
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.to_numpy(),
        feature_names=X_train.columns.tolist(),
        class_names=ACTIVITY_NAMES,
        mode="classification",
        discretize_continuous=True,
        random_state=random_state
    )

    # 예측된 클래스에 대한 로컬 설명 생성 (주요 10개 피처)
    exp = explainer.explain_instance(
        data_row=sample_values,
        predict_fn=model.predict_proba,
        num_features=10,
        labels=[pred_class_idx]
    )

    lime_rules = exp.as_list(label=pred_class_idx)

    print("\n" + "=" * 70)
    print(f" [5] LIME 설명 결과 상세 분석 (예측 클래스: {pred_activity})")
    print("=" * 70)
    print("  * 가중치(Weight) 부호 의미:")
    print("    - 양수(+): 해당 주성분(PC) 조건이 이 행동으로 예측하는 데 '기여/지지'함")
    print("    - 음수(-): 해당 주성분(PC) 조건이 이 행동으로 예측하는 것을 '방해/반대'함\n")

    pos_features = []
    neg_features = []

    for rule, weight in lime_rules:
        sign = "+" if weight >= 0 else "-"
        arrow = "▲ 기여(Positive)" if weight >= 0 else "▼ 반대(Negative)"
        print(f"  {sign} [{arrow:15s}] 가중치: {weight:+.4f} | 조건: {rule}")
        if weight >= 0:
            pos_features.append((rule, weight))
        else:
            neg_features.append((rule, weight))

    # 요약 자연어 해설
    print("\n" + "-" * 70)
    print(" [6] 결과 해석 및 결론 요약")
    print("-" * 70)
    print(f"  1. 모델은 이 샘플을 {pred_confidence*100:.1f}%의 높은 확신도로 '{pred_activity}'(으)로 분류했습니다.")
    if pred_class_idx == true_class_idx:
        print("     -> 실제 정답과 일치하여 올바르게 분류되었습니다.")
    else:
        print(f"     -> 실제 정답({true_activity})과 차이가 있어 오분류되었습니다.")

    if pos_features:
        top_pos = pos_features[0]
        print(f"  2. 가장 강력하게 해당 결정을 지지한 주성분 조건은 '{top_pos[0]}' (기여도: {top_pos[1]:+.4f}) 입니다.")
    if neg_features:
        top_neg = neg_features[0]
        print(f"  3. 반면 예측에 부정적 영향을 준 조건은 '{top_neg[0]}' (기여도: {top_neg[1]:+.4f}) 입니다.")
    print("  4. LIME은 국소적 선형 대리 모델(Surrogate)을 통해 고차원 PCA 공간에서도")
    print("     어떤 주성분의 값 범위가 최종 행동 판단을 유도했는지 명확한 설명 가능성을 제공합니다.")

    # 시각화 저장
    save_lime_visualization(
        exp=exp,
        prob_distribution=prob_distribution,
        pred_class_idx=pred_class_idx,
        true_class_idx=true_class_idx,
        sample_idx=sample_idx,
        lime_rules=lime_rules
    )

    # 인터랙티브 HTML 저장
    html_path = OUTPUT_DIR / f"lime_sample_{sample_idx}_report.html"
    exp.save_to_file(str(html_path))
    print(f"\n  >> LIME 인터랙티브 리포트 저장 완료: {html_path}")


def save_lime_visualization(exp, prob_distribution, pred_class_idx, true_class_idx, sample_idx, lime_rules):
    """
    클래스 예측 확률 및 LIME 피처 기여도 시각화 차트 생성 및 저장
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#FAFAFA")

    # 1. 클래스별 확률 분포 차트
    ax1 = axes[0]
    ax1.set_facecolor("#FFFFFF")
    colors = ["#4C72B0" if i != pred_class_idx else "#D95F02" for i in range(len(ACTIVITY_NAMES))]
    bars = ax1.barh(ACTIVITY_NAMES, prob_distribution, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_xlabel("예측 확률 (Probability)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"[샘플 #{sample_idx}] 클래스별 예측 확률 분포\n(실제: {ACTIVITY_NAMES[true_class_idx]} / 예측: {ACTIVITY_NAMES[pred_class_idx]})",
        fontsize=12, fontweight="bold", pad=12
    )
    ax1.set_xlim(0, 1.05)
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        if w > 0.01:
            ax1.text(w + 0.02, bar.get_y() + bar.get_height() / 2, f"{w*100:.1f}%",
                     va="center", ha="left", fontsize=10, fontweight="bold")

    # 2. LIME 기여도 바차트
    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF")
    rules = [r[0] for r in lime_rules][::-1]
    weights = [r[1] for r in lime_rules][::-1]
    rule_colors = ["#2CA02C" if w >= 0 else "#D62728" for w in weights]

    ax2.barh(rules, weights, color=rule_colors, edgecolor="black", alpha=0.85)
    ax2.axvline(0, color="black", linewidth=1.2, linestyle="-")
    ax2.set_xlabel(f"LIME 가중치 (클래스 '{ACTIVITY_NAMES[pred_class_idx]}' 지지 여부)", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"LIME 주요 주성분 기여도 (Top 10 Features)\n[초록: 지지(+), 빨강: 반대(-)]",
        fontsize=12, fontweight="bold", pad=12
    )
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"lime_sample_{sample_idx}_explanation.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  >> LIME 시각화 이미지 저장 완료: {chart_path}")


def main():
    # 1. 데이터 로드 및 분할
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data(test_size=0.2, random_state=42)

    # 2. XGBoost 모델 학습
    model = train_xgboost(X_train, y_train, random_state=42)

    # 3. 모델 평가
    evaluate_model(model, X_test, y_test)

    # 4. 임의의 테스트 샘플 선정 및 LIME XAI 적용 및 설명
    # sample_idx=None으로 설정하면 난수 시드에 의해 자동으로 임의의 테스트 샘플을 선택합니다.
    explain_sample_with_lime(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        sample_idx=None,  # 또는 특정 인덱스 (예: 42, 100 등)
        random_state=42
    )


if __name__ == "__main__":
    main()
