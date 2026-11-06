"""
K=3 vs K=5 실루엣 계수 비교
- Mall_Customers.csv의 Annual Income, Spending Score 2변수를 사용
- Ward 연결법 기반 계층적 군집화(AgglomerativeClustering)로 K=3, K=5 각각 군집화 후 실루엣 계수 비교
"""

from pathlib import Path

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

DATA_PATH = Path(__file__).resolve().parent / "data" / "Mall_Customers.csv"

df = pd.read_csv(DATA_PATH)
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

results = {}
for k in (3, 5):
    model = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
    labels = model.fit_predict(X)
    results[k] = silhouette_score(X, labels)

for k, score in results.items():
    print(f"K={k} 실루엣 계수: {score:.4f}")

best_k = max(results, key=results.get)
print(f"\n>> 더 좋은 K: {best_k} (실루엣 계수 {results[best_k]:.4f})")
