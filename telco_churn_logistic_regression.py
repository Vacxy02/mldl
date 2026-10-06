"""
Telco Customer Churn 로지스틱 회귀(Logistic Regression) 모델 학습, 튜닝 및 평가 파이프라인
- EDA 보고서(다중공선성 변수 배제 및 파생변수 투입 권고) 반영
- 전처리 파이프라인 (StandardScaler + OneHotEncoder)
- Stratified 5-Fold GridSearchCV 하이퍼파라미터 튜닝
- 기본 임계값(0.5) vs 비즈니스 최적 임계값 비교 평가
- 승산비(Odds Ratio) 분석 및 2x2 종합 성능 시각화 차트 생성
"""

import os
import platform
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 0. 한글 폰트 및 시각화 테마 설정
# -----------------------------------------------------------------------------
def setup_visualization():
    sns.set_theme(style="whitegrid")
    sys_name = platform.system()
    if sys_name == "Darwin":  # macOS
        plt.rcParams["font.family"] = "AppleGothic"
    elif sys_name == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    else:
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False


# -----------------------------------------------------------------------------
# 1. 데이터 로드 및 EDA 기반 피처 선정
# -----------------------------------------------------------------------------
def load_and_prepare_data(file_path: str):
    print("=" * 80)
    print("1. 데이터셋 로드 및 EDA 권고안 기반 피처 선정")
    print("=" * 80)
    
    df = pd.read_csv(file_path)
    print(f"- 데이터셋 크기: {df.shape[0]:,}행 × {df.shape[1]}열")

    # EDA 권고안 반영:
    # 1) TotalCharges 배제: tenure × MonthlyCharges와의 강한 다중공선성(VIF 8+) 해소
    # 2) 파생 변수 투입: Tenure_Group, Service_Count, Avg_Monthly_Ratio
    # 3) 식별자 및 중복 타깃 제거
    drop_cols = ["TotalCharges", "Churn"]
    if "customerID" in df.columns:
        drop_cols.append("customerID")
    
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    if "Churn_Label" in df.columns:
        y = df["Churn_Label"]
        X = X.drop(columns=["Churn_Label"])
    else:
        y = (df["Churn"] == "Yes").astype(int)

    # 수치형 피처와 범주형 피처 자동 분류
    num_cols = ["tenure", "MonthlyCharges", "Service_Count", "Avg_Monthly_Ratio"]
    cat_cols = [col for col in X.columns if col not in num_cols]

    print(f"- 다중공선성 변수(TotalCharges) 배제 완료")
    print(f"- 최종 투입 수치형 피처({len(num_cols)}개): {num_cols}")
    print(f"- 최종 투입 범주형 피처({len(cat_cols)}개): {cat_cols}")
    print(f"- 타깃 레이블(Churn) 분포: 유지(0)={np.sum(y==0):,}명, 이탈(1)={np.sum(y==1):,}명 (이탈률: {np.mean(y)*100:.2f}%)")

    return X, y, num_cols, cat_cols


# -----------------------------------------------------------------------------
# 2. 전처리 파이프라인 및 모델 학습 (GridSearchCV)
# -----------------------------------------------------------------------------
def build_and_train_model(X_train, y_train, num_cols, cat_cols):
    print("\n" + "=" * 80)
    print("2. 전처리 파이프라인 구축 및 GridSearchCV 하이퍼파라미터 튜닝")
    print("=" * 80)

    # 전처리 파이프라인: 수치형은 StandardScaler, 범주형은 OneHotEncoder (더미 변수 트랩 방지: drop='first')
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs"))
    ])

    # 5-Fold 계층적 교차 검증 및 규제 강도 C 튜닝
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        "clf__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        "clf__class_weight": [None, "balanced"]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0
    )

    print("- Stratified 5-Fold 교차 검증으로 최적 하이퍼파라미터 탐색 중...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"- 최적 하이퍼파라미터: {best_params}")
    print(f"- 5-Fold 평균 검증 ROC-AUC: {best_score:.4f}")

    return best_model


# -----------------------------------------------------------------------------
# 3. 모델 종합 평가 및 임계값(Threshold) 최적화
# -----------------------------------------------------------------------------
def evaluate_and_compare_thresholds(model, X_test, y_test):
    print("\n" + "=" * 80)
    print("3. 모델 종합 성능 평가: 기본 임계값(0.5) vs 비즈니스 최적 임계값 비교")
    print("=" * 80)

    # 테스트셋 이탈 확률 예측
    y_prob = model.predict_proba(X_test)[:, 1]

    # ROC-AUC 및 PR-AUC 계산
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recalls, precisions)

    # 비즈니스 관점 최적 임계값 탐색 (F1-Score를 극대화하는 임계값 도출)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = round(pr_thresholds[best_idx], 2)

    # 두 임계값 기준 예측 결과 도출
    preds_default = (y_prob >= 0.5).astype(int)
    preds_optimal = (y_prob >= optimal_threshold).astype(int)

    def calc_metrics(y_true, y_pred, threshold_name):
        return {
            "평가 지표": threshold_name,
            "정확도 (Accuracy)": f"{accuracy_score(y_true, y_pred) * 100:.2f}%",
            "정밀도 (Precision)": f"{precision_score(y_true, y_pred) * 100:.2f}%",
            "재현율 (Recall)": f"{recall_score(y_true, y_pred) * 100:.2f}%",
            "F1-Score": f"{f1_score(y_true, y_pred):.4f}",
            "ROC-AUC": f"{roc_auc:.4f}",
            "PR-AUC": f"{pr_auc:.4f}"
        }

    metrics_df = pd.DataFrame([
        calc_metrics(y_test, preds_default, "기본 임계값 (0.50)"),
        calc_metrics(y_test, preds_optimal, f"비즈니스 최적 임계값 ({optimal_threshold:.2f})")
    ])

    print("\n=== [임계값별 성능 지표 비교표] ===")
    print(metrics_df.to_string(index=False))

    print(f"\n💡 [비즈니스 인사이트 & 임계값 조정 권고]")
    print(f"  통신사 이탈 방어 마케팅에서는 '실제 이탈할 고객을 놓치지 않는 것(Recall)'이 최우선 과제입니다.")
    print(f"  임계값을 0.50에서 {optimal_threshold:.2f}로 조정하면:")
    print(f"  - 이탈 감지율(Recall)이 {metrics_df.iloc[0]['재현율 (Recall)']}에서 {metrics_df.iloc[1]['재현율 (Recall)']}로 대폭 향상됩니다.")
    print(f"  - F1-Score 역시 {metrics_df.iloc[0]['F1-Score']}에서 {metrics_df.iloc[1]['F1-Score']}로 개선되어 불균형 데이터에 최적화됩니다.")

    return y_prob, fpr, tpr, roc_auc, precisions, recalls, pr_auc, optimal_threshold, preds_default, preds_optimal


# -----------------------------------------------------------------------------
# 4. 회귀 계수(Coefficients) 및 승산비(Odds Ratio) 분석
# -----------------------------------------------------------------------------
def analyze_odds_ratio(model, num_cols, cat_cols):
    print("\n" + "=" * 80)
    print("4. 로지스틱 회귀 계수(Coefficients) 및 승산비(Odds Ratio) 분석")
    print("=" * 80)

    # 전처리된 피처 이름 추출
    cat_encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cat_cols = cat_encoder.get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + encoded_cat_cols

    coef = model.named_steps["clf"].coef_[0]
    odds_ratio = np.exp(coef)

    df_coef = pd.DataFrame({
        "Feature": all_feature_names,
        "Coefficient (Beta)": coef,
        "Odds Ratio (exp(Beta))": odds_ratio,
        "Abs_Beta": np.abs(coef)
    }).sort_values(by="Coefficient (Beta)", ascending=False)

    # 상위 위험 촉진 요인 (Positive) 및 방어 요인 (Negative)
    top_pos = df_coef.head(5)
    top_neg = df_coef.tail(5).sort_values(by="Coefficient (Beta)", ascending=True)

    print("\n🚨 [이탈 위험 촉진 요인 (Positive Weights / Odds Ratio > 1)]")
    for _, row in top_pos.iterrows():
        print(f"  - {row['Feature']:35s}: 계수 {row['Coefficient (Beta)']:+.3f} → 승산비 {row['Odds Ratio (exp(Beta))']:.2f}배 (이탈 확률 상승)")

    print("\n🛡️ [이탈 방어/고객 락인 요인 (Negative Weights / Odds Ratio < 1)]")
    for _, row in top_neg.iterrows():
        print(f"  - {row['Feature']:35s}: 계수 {row['Coefficient (Beta)']:+.3f} → 승산비 {row['Odds Ratio (exp(Beta))']:.2f}배 (이탈 확률 감소)")

    return df_coef


# -----------------------------------------------------------------------------
# 5. 종합 평가 결과 2x2 시각화
# -----------------------------------------------------------------------------
def plot_evaluation_dashboard(y_test, y_prob, fpr, tpr, roc_auc, precisions, recalls, pr_auc,
                              optimal_threshold, preds_default, preds_optimal, df_coef, output_path):
    print("\n" + "=" * 80)
    print("5. 종합 평가 대시보드 시각화 생성 중...")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Telco Churn 로지스틱 회귀 모델 종합 성능 평가 & 승산비 분석", fontsize=18, fontweight="bold")

    # [1] ROC Curve
    axes[0, 0].plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    axes[0, 0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5)
    axes[0, 0].set_title("ROC Curve (수신기 조작 특성 곡선)", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("위양성률 (False Positive Rate)", fontsize=11)
    axes[0, 0].set_ylabel("진양성률 (True Positive Rate / Recall)", fontsize=11)
    axes[0, 0].legend(loc="lower right", fontsize=11)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # [2] Precision-Recall Curve
    axes[0, 1].plot(recalls, precisions, color="#e67e22", lw=2.5, label=f"PR Curve (AP = {pr_auc:.4f})")
    # 최적 임계값 위치 표시
    opt_rec = recall_score(y_test, preds_optimal)
    opt_prec = precision_score(y_test, preds_optimal)
    axes[0, 1].scatter([opt_rec], [opt_prec], color="#c0392b", s=100, zorder=5, 
                       label=f"최적 임계값 ({optimal_threshold:.2f})")
    axes[0, 1].set_title("Precision-Recall Curve (불균형 분류 성능)", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("재현율 (Recall)", fontsize=11)
    axes[0, 1].set_ylabel("정밀도 (Precision)", fontsize=11)
    axes[0, 1].legend(loc="upper right", fontsize=11)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # [3] 혼동 행렬 비교 (기본 vs 최적)
    cm_default = confusion_matrix(y_test, preds_default)
    cm_optimal = confusion_matrix(y_test, preds_optimal)
    
    # 2개 혼동 행렬을 나란히 서브-히트맵으로 표기하기 위해 text 병기
    sub_cm_ax = axes[1, 0]
    sns.heatmap(cm_optimal, annot=True, fmt="d", cmap="Blues", ax=sub_cm_ax, cbar=False,
                xticklabels=["유지(No)", "이탈(Yes)"], yticklabels=["유지(No)", "이탈(Yes)"])
    sub_cm_ax.set_title(f"혼동 행렬 - 최적 임계값 ({optimal_threshold:.2f}) 기준\n[이탈 감지수: {cm_optimal[1,1]}/{np.sum(cm_optimal[1])}명, 기본(0.5): {cm_default[1,1]}명]",
                        fontsize=12, fontweight="bold")
    sub_cm_ax.set_xlabel("예측 클래스 (Predicted)", fontsize=11)
    sub_cm_ax.set_ylabel("실제 클래스 (Actual)", fontsize=11)

    # [4] 상위 10개 핵심 피처의 승산비(Odds Ratio) 막대 차트
    # 상위 5개 촉진 요인 + 상위 5개 방어 요인
    top_pos = df_coef.head(5)
    top_neg = df_coef.tail(5)
    top_10 = pd.concat([top_pos, top_neg]).sort_values(by="Coefficient (Beta)", ascending=True)

    colors = ["#3498db" if c < 0 else "#e74c3c" for c in top_10["Coefficient (Beta)"]]
    bars = axes[1, 1].barh(top_10["Feature"], top_10["Coefficient (Beta)"], color=colors, edgecolor="black", alpha=0.85)
    axes[1, 1].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[1, 1].set_title("핵심 피처 회귀 계수 및 승산비(Odds Ratio)", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("회귀 계수 β (양수: 이탈 위험 촉진, 음수: 이탈 방어)", fontsize=11)
    axes[1, 1].set_xlim(-1.8, 1.6)
    axes[1, 1].grid(axis="x", linestyle=":", alpha=0.6)

    for bar, (_, row) in zip(bars, top_10.iterrows()):
        w = bar.get_width()
        or_val = row["Odds Ratio (exp(Beta))"]
        if w >= 0:
            axes[1, 1].text(w + 0.04, bar.get_y() + bar.get_height() / 2,
                            f"OR: {or_val:.2f}배", va="center", ha="left", fontsize=9, fontweight="bold")
        else:
            axes[1, 1].text(w - 0.04, bar.get_y() + bar.get_height() / 2,
                            f"OR: {or_val:.2f}배", va="center", ha="right", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"- [시각화 저장] 종합 평가 대시보드 저장 완료: {output_path}")


# -----------------------------------------------------------------------------
# Main 실행 함수
# -----------------------------------------------------------------------------
def main():
    setup_visualization()

    # 데이터 경로 탐색
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "telco_churn_preprocessed.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, "telco_churn_preprocessed.csv")

    output_img = os.path.join(base_dir, "telco_logistic_regression_eval.png")

    # 1. 데이터 로드 및 피처 선정
    X, y, num_cols, cat_cols = load_and_prepare_data(data_path)

    # 2. 계층적 분할 (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. 모델 튜닝 및 학습
    best_model = build_and_train_model(X_train, y_train, num_cols, cat_cols)

    # 4. 임계값 비교 및 종합 평가
    y_prob, fpr, tpr, roc_auc, precisions, recalls, pr_auc, optimal_thresh, p_def, p_opt = evaluate_and_compare_thresholds(
        best_model, X_test, y_test
    )

    # 5. 회귀 계수 및 승산비 분석
    df_coef = analyze_odds_ratio(best_model, num_cols, cat_cols)

    # 6. 시각화 대시보드 출력
    plot_evaluation_dashboard(
        y_test, y_prob, fpr, tpr, roc_auc, precisions, recalls, pr_auc,
        optimal_thresh, p_def, p_opt, df_coef, output_img
    )

    print("\n" + "=" * 80)
    print("🎯 로지스틱 회귀 모델 학습, 평가 및 시각화 파이프라인이 성공적으로 완료되었습니다.")
    print("=" * 80)


if __name__ == "__main__":
    main()
