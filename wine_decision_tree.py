"""
와인(Wine) 데이터셋 기반 의사결정나무(Decision Tree) 학습, 평가 및 시각화
- GridSearchCV 하이퍼파라미터 튜닝 (과적합 제어: max_depth, min_samples_split 등)
- 모델 평가 (정확도, F1-Score, 분류 보고서, 혼동 행렬)
- Graphviz를 이용한 의사결정나무 구조 시각화 (PNG 저장)
- 피처 중심 시각화 (특성 중요도 Feature Importance + 상위 2개 특성 결정 경계 Decision Boundary)
"""

import os
import platform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import graphviz
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# ---------------------------------------------------------
# 0. 한글 폰트 및 시각화 테마 설정
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 1. 데이터 로드 및 분할
# ---------------------------------------------------------
def load_and_split_data():
    print("=" * 75)
    print("1. 와인(Wine) 데이터셋 로드 및 분할")
    print("=" * 75)

    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")
    target_names = wine.target_names

    print(f"- 샘플 수: {X.shape[0]}개, 특성 수: {X.shape[1]}개")
    print(f"- 타깃 클래스: {list(target_names)} (0, 1, 2)")

    # 계층적 샘플링(stratify=y)으로 Train(75%):Test(25%) 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"- Train 샘플 수: {len(X_train)}개, Test 샘플 수: {len(X_test)}개")

    return wine, X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# 2. 모델 학습 및 하이퍼파라미터 튜닝 (GridSearchCV)
# ---------------------------------------------------------
def train_decision_tree(X_train, y_train):
    print("\n" + "=" * 75)
    print("2. 의사결정나무(Decision Tree) 하이퍼파라미터 튜닝 및 학습")
    print("=" * 75)

    # 과적합 제어를 위한 파라미터 그리드
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 4, 5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dt = DecisionTreeClassifier(random_state=42)

    grid_search = GridSearchCV(
        dt,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f"- 최적 하이퍼파라미터: {grid_search.best_params_}")
    print(f"- 5-Fold 교차 검증 최고 정확도: {grid_search.best_score_ * 100:.2f}%")

    return best_model


# ---------------------------------------------------------
# 3. 모델 성능 평가
# ---------------------------------------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, target_names):
    print("\n" + "=" * 75)
    print("3. 모델 종합 성능 평가")
    print("=" * 75)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    y_pred = model.predict(X_test)

    print(f"- 훈련 세트(Train) 정확도: {train_acc * 100:.2f}%")
    print(f"- 테스트 세트(Test) 정확도: {test_acc * 100:.2f}%")
    print(f"  -> 과적합 여부: {'적절한 일반화 성능 달성' if abs(train_acc - test_acc) < 0.1 else '약간의 과적합 발생'}")

    print("\n[분류 보고서 (Classification Report)]")
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    return cm, y_pred


# ---------------------------------------------------------
# 4. Graphviz를 이용한 트리 시각화
# ---------------------------------------------------------
def visualize_tree_graphviz(model, wine, output_filename="wine_decision_tree"):
    print("\n" + "=" * 75)
    print("4. Graphviz를 이용한 의사결정나무 시각화 생성")
    print("=" * 75)

    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=wine.feature_names,
        class_names=wine.target_names,
        filled=True,
        rounded=True,
        special_characters=True,
        precision=2
    )

    graph = graphviz.Source(dot_data)
    # png 파일로 렌더링
    graph.render(output_filename, format="png", cleanup=True)
    out_img = f"{output_filename}.png"
    print(f"- Graphviz 의사결정나무 다이어그램 저장 완료: {out_img}")
    return out_img


# ---------------------------------------------------------
# 5. 피처 중심 시각화 (특성 중요도 + 결정 경계 + 혼동 행렬)
# ---------------------------------------------------------
def visualize_feature_analysis(model, wine, X_train, y_train, cm, target_names, output_filename="wine_tree_feature_analysis.png"):
    print("\n" + "=" * 75)
    print("5. 피처 중심 분석 및 결정 경계 시각화")
    print("=" * 75)

    feature_names = wine.feature_names
    importances = model.feature_importances_

    # 특성 중요도 데이터프레임
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    fig.suptitle("와인 데이터셋 Decision Tree 피처 중심 종합 분석", fontsize=16, fontweight="bold")

    # [1] 특성 중요도 (Feature Importance) 가로 막대 차트
    # 0보다 큰 피처들을 위주로 시각화
    colors = ["#3498db" if imp < 0.2 else "#e74c3c" for imp in df_imp["Importance"]]
    bars = axes[0].barh(df_imp["Feature"], df_imp["Importance"], color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("특성 중요도 (Feature Importance)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("지니 불순도 감소 기여도 (Importance)", fontsize=11)
    axes[0].set_xlim(0, max(df_imp["Importance"]) * 1.15)
    axes[0].grid(axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        if w > 0:
            axes[0].text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}",
                         va="center", ha="left", fontsize=9, fontweight="bold")

    # [2] 혼동 행렬 (Confusion Matrix)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=axes[1],
        cbar=False,
        annot_kws={"size": 13, "weight": "bold"}
    )
    axes[1].set_title("혼동 행렬 (Confusion Matrix)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("예측 품종 (Predicted)", fontsize=11)
    axes[1].set_ylabel("실제 품종 (Actual)", fontsize=11)

    # [3] 최상위 2개 핵심 피처의 2D 결정 경계 (Decision Boundary)
    # 가장 중요한 상위 2개 피처 선정
    top_2_features = df_imp.sort_values(by="Importance", ascending=False)["Feature"].head(2).tolist()
    feat_x, feat_y = top_2_features[0], top_2_features[1]
    print(f"- 최상위 2개 피처 선정: X축='{feat_x}', Y축='{feat_y}'")

    # 2개 피처만으로 시각화용 간이 결정 트리 학습
    clf_2d = DecisionTreeClassifier(
        max_depth=model.max_depth,
        min_samples_split=model.min_samples_split,
        min_samples_leaf=model.min_samples_leaf,
        random_state=42
    )
    X_train_2d = X_train[[feat_x, feat_y]].values
    clf_2d.fit(X_train_2d, y_train.values)

    # 격자(Grid) 생성
    x_min, x_max = X_train_2d[:, 0].min() - 0.5, X_train_2d[:, 0].max() + 0.5
    y_min, y_max = X_train_2d[:, 1].min() - 0.5, X_train_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 등고선 채우기로 결정 영역 시각화
    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ["#d63031", "#00b894", "#0984e3"]

    axes[2].contourf(xx, yy, Z, alpha=0.35, cmap=cmap_light)

    # 데이터 포인트 산점도
    for cl_idx, cl_name in enumerate(target_names):
        idx = y_train.values == cl_idx
        axes[2].scatter(
            X_train_2d[idx, 0], X_train_2d[idx, 1],
            c=cmap_bold[cl_idx], label=cl_name, edgecolor="k", s=45, alpha=0.9
        )

    axes[2].set_title(f"핵심 2개 피처의 결정 경계 (Decision Boundary)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel(feat_x, fontsize=11)
    axes[2].set_ylabel(feat_y, fontsize=11)
    axes[2].legend(loc="upper right", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"- 피처 중심 종합 분석 차트 저장 완료: {output_filename}")


# ---------------------------------------------------------
# Main 실행 함수
# ---------------------------------------------------------
def main():
    setup_visualization()

    # 1. 데이터 로드
    wine, X_train, X_test, y_train, y_test = load_and_split_data()

    # 2. 모델 학습 및 하이퍼파라미터 튜닝
    best_model = train_decision_tree(X_train, y_train)

    # 3. 모델 평가
    cm, y_pred = evaluate_model(best_model, X_train, y_train, X_test, y_test, wine.target_names)

    # 4. Graphviz 의사결정나무 시각화
    visualize_tree_graphviz(best_model, wine, output_filename="wine_decision_tree")

    # 5. 피처 중심 시각화 (특성 중요도 + 혼동 행렬 + 2D 결정 경계)
    visualize_feature_analysis(best_model, wine, X_train, y_train, cm, wine.target_names,
                               output_filename="wine_tree_feature_analysis.png")

    print("\n" + "=" * 75)
    print("🎯 모든 디시전트리 학습, 평가 및 시각화가 성공적으로 완료되었습니다!")
    print("=" * 75)


if __name__ == "__main__":
    main()
