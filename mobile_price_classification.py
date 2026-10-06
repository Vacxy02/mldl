import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def main():
    # 1. 데이터 불러오기
    train_path = 'mobile_price_category_train.csv'
    test_path = 'mobile_price_category_test.csv'

    print(">> 데이터 로딩 중...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 타깃 컬럼 및 피처 분리
    target_col = 'price_range'
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # Test 데이터셋 전처리 (id 컬럼 보존 및 피처 맞춤)
    if 'id' in test_df.columns:
        test_ids = test_df['id']
        X_test = test_df.drop(columns=['id'])
    else:
        test_ids = test_df.index
        X_test = test_df.copy()

    # 열 순서 일치 확인
    X_test = X_test[X.columns]

    print(f"Train Shape: {X.shape}, Target Distribution:\n{y.value_counts().sort_index()}")
    print(f"Test Shape: {X_test.shape}")

    # 2. 피처 스케일링 (수치형 피처 정규화)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # 3. 모델 교차 검증 (Stratified K-Fold)
    print("\n>> 5-Fold 교차 검증 진행...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)

    rf_scores = cross_val_score(rf_model, X_scaled, y, cv=cv, scoring='accuracy')
    gb_scores = cross_val_score(gb_model, X_scaled, y, cv=cv, scoring='accuracy')

    print(f"Random Forest 평균 정확도(Accuracy): {rf_scores.mean():.4f} (±{rf_scores.std():.4f})")
    print(f"Gradient Boosting 평균 정확도(Accuracy): {gb_scores.mean():.4f} (±{gb_scores.std():.4f})")

    # 4. 검증 데이터셋 분할 및 상세 평가 (Random Forest 기준)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n>> 모델 학습 및 Validation 평가 (Random Forest)...")
    rf_model.fit(X_train, y_train)
    val_preds = rf_model.predict(X_val)

    print(f"검증셋 정확도: {accuracy_score(y_val, val_preds):.4f}")
    print("\n[Classification Report]")
    print(classification_report(y_val, val_preds, target_names=['Class 0 (Low)', 'Class 1 (Medium)', 'Class 2 (High)', 'Class 3 (Very High)']))

    # 5. 특성 중요도(Feature Importance) 출력
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print(">> 상위 5개 중요 피처:")
    print(importance_df.head(5).to_string(index=False))

    # 6. 전체 Train 데이터로 최종 재학습 후 Test 예측
    print("\n>> 전체 Train 데이터셋으로 최종 모델 학습 및 Test 예측...")
    rf_model.fit(X_scaled, y)
    test_preds = rf_model.predict(X_test_scaled)

    # 7. 예측 결과 CSV 파일 저장
    submission = pd.DataFrame({
        'id': test_ids,
        'price_range': test_preds
    })
    output_filename = 'submission.csv'
    submission.to_csv(output_filename, index=False)
    print(f">> 예측 결과가 '{output_filename}'으로 성공적으로 저장되었습니다.")
    print(submission.head())

if __name__ == '__main__':
    main()
