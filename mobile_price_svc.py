import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def preprocess_and_feature_engineering(train_df, test_df):
    """
    1. 비정상값(0값) 보정
    2. 파생 변수 생성
    """
    train = train_df.copy()
    test = test_df.copy()

    # 1. px_height와 sc_w의 0값을 Train의 중앙값(median)으로 보정
    for col in ['px_height', 'sc_w']:
        # Train의 0이 아닌 값들의 중앙값 계산
        median_val = train.loc[train[col] > 0, col].median()
        train[col] = train[col].replace(0, median_val)
        test[col] = test[col].replace(0, median_val)

    # 2. 파생 변수 생성
    for df in [train, test]:
        # 총 해상도 화소수 (px_area)
        df['px_area'] = df['px_height'] * df['px_width']
        # 화면 면적 (sc_area, cm^2)
        df['sc_area'] = df['sc_h'] * df['sc_w']
        # 배터리 사용 효율 (battery_power / talk_time)
        df['battery_efficiency'] = df['battery_power'] / (df['talk_time'] + 1e-5)

    return train, test

def main():
    print("=" * 65)
    print(">> 1. 데이터 로드 및 전처리 시작...")
    print("=" * 65)

    train_df = pd.read_csv('mobile_price_category_train.csv')
    test_df = pd.read_csv('mobile_price_category_test.csv')

    target_col = 'price_range'
    y = train_df[target_col]
    X_raw = train_df.drop(columns=[target_col])

    # Test의 id 컬럼 보존
    if 'id' in test_df.columns:
        test_ids = test_df['id']
        X_test_raw = test_df.drop(columns=['id'])
    else:
        test_ids = test_df.index
        X_test_raw = test_df.copy()

    # 열 순서 맞춤
    X_test_raw = X_test_raw[X_raw.columns]

    # 전처리 및 파생 변수 생성 적용
    X_fe, X_test_fe = preprocess_and_feature_engineering(X_raw, X_test_raw)
    print(f"- 원본 피처 개수 : {X_raw.shape[1]}개")
    print(f"- 파생 피처 추가 후 : {X_fe.shape[1]}개 ({list(X_fe.columns[-3:])} 추가)")

    # 2. Train / Validation 분할 (검증 및 최종 모델 평가용)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_fe, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 특성 스케일링 (StandardScaler)
    print("\n" + "=" * 65)
    print(">> 2. 특성 스케일링 (StandardScaler) 적용...")
    print("=" * 65)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    
    # 전체 Train셋 스케일러 및 Test셋 스케일링
    full_scaler = StandardScaler()
    X_full_scaled = full_scaler.fit_transform(X_fe)
    X_test_scaled = full_scaler.transform(X_test_fe)

    # 4. 하이퍼파라미터 그리드 서치 (GridSearchCV with StratifiedKFold)
    print("\n" + "=" * 65)
    print(">> 3. SVC 하이퍼파라미터 튜닝 (GridSearchCV)...")
    print("=" * 65)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = [
        # Linear 커널 탐색
        {
            'kernel': ['linear'],
            'C': [0.1, 1, 10, 50],
            'decision_function_shape': ['ovo']
        },
        # RBF 비선형 커널 탐색
        {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10, 50],
            'gamma': ['scale', 0.001, 0.01, 0.1],
            'decision_function_shape': ['ovo']
        }
    ]

    base_svc = SVC(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_svc,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    print(f"\n[최적 하이퍼파라미터]: {grid_search.best_params_}")
    print(f"[교차 검증 최고 정확도 (CV Score)]: {grid_search.best_score_:.4f}")

    # 5. 검증 데이터셋(Validation Set) 상세 평가
    print("\n" + "=" * 65)
    print(">> 4. Validation 데이터셋 상세 성능 평가...")
    print("=" * 65)

    best_model = grid_search.best_estimator_
    val_preds = best_model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_preds)

    print(f"Validation 정확도: {val_acc:.4f}\n")
    print("[Classification Report]")
    print(classification_report(
        y_val, val_preds,
        target_names=['Class 0 (Low)', 'Class 1 (Medium)', 'Class 2 (High)', 'Class 3 (Very High)']
    ))

    print("[Confusion Matrix]")
    print(confusion_matrix(y_val, val_preds))

    # 6. 전체 데이터로 최종 학습 및 Test 예측값 저장
    print("\n" + "=" * 65)
    print(">> 5. 전체 데이터 재학습 및 Test 예측 파일 생성...")
    print("=" * 65)

    final_svc = SVC(**grid_search.best_params_, random_state=42)
    final_svc.fit(X_full_scaled, y)
    test_preds = final_svc.predict(X_test_scaled)

    submission = pd.DataFrame({
        'id': test_ids,
        'price_range': test_preds
    })
    output_filename = 'submission_svc.csv'
    submission.to_csv(output_filename, index=False)
    print(f">> 예측 결과가 '{output_filename}'으로 저장되었습니다.")
    print(submission.head())

if __name__ == '__main__':
    main()
