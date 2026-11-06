import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/two_spirals_noisy_dense.csv')

fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(df['x'], df['y'], c=df['cluster'], cmap='tab10', s=15)

ax.set_title('Two Spirals (colored by cluster)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')

legend = ax.legend(*scatter.legend_elements(), title='cluster')
ax.add_artist(legend)

plt.tight_layout()
plt.show()
