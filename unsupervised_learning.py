"""
=============================================================
BÀI KIỂM TRA SỐ 3: HỌC KHÔNG GIÁM SÁT
Phân cụm NYC Airbnb Data với K-Means và DBSCAN
=============================================================

Dataset: New York City Airbnb Open Data
Nguồn: Kaggle / Inside Airbnb
Số mẫu: 20,758 | Số features: 28 (sau xử lý)

Yêu cầu cài đặt:
    pip install pandas numpy matplotlib seaborn scikit-learn reportlab
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend không cần GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score)
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 1: ĐỌC VÀ KHÁM PHÁ DỮ LIỆU
# ─────────────────────────────────────────────────────────────────────────────

# Đọc cả 2 file: dữ liệu gốc và đã xử lý
pre  = pd.read_csv("preprocess_data.csv")
post = pd.read_csv("postprocess_data.csv")

print("=" * 60)
print("1. KHÁM PHÁ DỮ LIỆU")
print("=" * 60)
print(f"Preprocess data shape:  {pre.shape}")
print(f"Postprocess data shape: {post.shape}")

# Kiểm tra missing values trong dữ liệu gốc
print(f"\nMissing values (preprocess):\n{pre.isnull().sum()[pre.isnull().sum()>0]}")
print(f"\nMissing values (postprocess): {post.isnull().sum().sum()} (đã xử lý sạch)")

# Chuyển cột dạng object sang numeric
for col in ['baths', 'bedrooms', 'rating']:
    if col in pre.columns:
        pre[col] = pd.to_numeric(pre[col], errors='coerce')

print(f"\nMô tả thống kê (dữ liệu gốc):")
print(pre.describe().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 2: CHUẨN BỊ DỮ LIỆU CHO PHÂN CỤM
# ─────────────────────────────────────────────────────────────────────────────

# Dùng postprocess_data.csv (đã StandardScaler + One-Hot + Feature Engineering)
# Loại bỏ cột 'const' nếu có
X = post.drop(columns=['const'], errors='ignore').values
print(f"\nFeature matrix X: {X.shape}")
print(f"Columns: {list(post.drop(columns=['const'], errors='ignore').columns)}")

import os
os.makedirs("charts", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 3: TRỰC QUAN HÓA DỮ LIỆU BAN ĐẦU
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("3. TRỰC QUAN HÓA DỮ LIỆU")
print("="*60)

cont_cols = ['price','minimum_nights','number_of_reviews',
             'reviews_per_month','availability_365','rating']
cont_cols = [c for c in cont_cols if c in pre.columns]

# 3.1 Histogram
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(cont_cols):
    d = pd.to_numeric(pre[col], errors='coerce').dropna()
    axes[i].hist(d, bins=30, color='steelblue', alpha=0.75, edgecolor='white')
    axes[i].set_title(col)
plt.suptitle('Phân phối thuộc tính liên tục', fontweight='bold')
plt.tight_layout(); plt.savefig("charts/01_histograms.png")
plt.close(); print("✓ Saved: charts/01_histograms.png")

# 3.2 Boxplot
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(cont_cols):
    d = pd.to_numeric(pre[col], errors='coerce').dropna()
    axes[i].boxplot(d, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    axes[i].set_title(col)
plt.suptitle('Boxplot thuộc tính liên tục', fontweight='bold')
plt.tight_layout(); plt.savefig("charts/02_boxplots.png")
plt.close(); print("✓ Saved: charts/02_boxplots.png")

# 3.3 Bar chart (categorical)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
if 'room_type' in pre.columns:
    pre['room_type'].value_counts().plot(kind='bar', ax=axes[0], color='coral', edgecolor='k')
    axes[0].set_title('Room Type Distribution'); axes[0].tick_params(axis='x', rotation=30)
if 'neighbourhood_group' in pre.columns:
    pre['neighbourhood_group'].value_counts().plot(kind='bar', ax=axes[1], color='mediumpurple', edgecolor='k')
    axes[1].set_title('Borough Distribution'); axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout(); plt.savefig("charts/03_categorical.png")
plt.close(); print("✓ Saved: charts/03_categorical.png")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 4: K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("4. K-MEANS CLUSTERING")
print("="*60)

# 4.1 Elbow Method + Silhouette Score để chọn k tối ưu
print("Đang tính Elbow + Silhouette (k=2..10)...")
np.random.seed(42)
idx_sub = np.random.choice(len(X), 5000, replace=False)  # Dùng 5000 mẫu cho tốc độ
X_sub = X[idx_sub]

sse, silhouettes = [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
    labels = km.fit_predict(X_sub)
    sse.append(km.inertia_)
    silhouettes.append(silhouette_score(X_sub, labels))
    print(f"  k={k}: SSE={km.inertia_:.0f}, Silhouette={silhouettes[-1]:.4f}")

# Tìm k tốt nhất theo Silhouette Score
best_k = list(K_range)[np.argmax(silhouettes)]
print(f"\n→ k tối ưu (Silhouette cao nhất): k = {best_k}")

# Vẽ biểu đồ Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_range), sse, 'bo-', lw=2, ms=7)
axes[0].set(xlabel='Số cụm k', ylabel='SSE (Inertia)', title='Elbow Method')
axes[0].grid(alpha=0.3)

axes[1].plot(list(K_range), silhouettes, 'rs-', lw=2, ms=7)
axes[1].axvline(best_k, color='green', ls='--', label=f'k={best_k} (tốt nhất)')
axes[1].set(xlabel='Số cụm k', ylabel='Silhouette Score', title='Silhouette Score vs k')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.savefig("charts/04_elbow_silhouette.png")
plt.close(); print("✓ Saved: charts/04_elbow_silhouette.png")

# 4.2 Chạy K-Means với k tối ưu trên toàn bộ dữ liệu
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_labels = km_final.fit_predict(X)

# Tính các chỉ số đánh giá
km_sil = silhouette_score(X, km_labels, sample_size=3000, random_state=42)
km_dbi = davies_bouldin_score(X, km_labels)
km_chi = calinski_harabasz_score(X, km_labels)

print(f"\nK-Means (k={best_k}) - Kết quả đánh giá:")
print(f"  Silhouette Score:       {km_sil:.4f}")
print(f"  Davies-Bouldin Index:   {km_dbi:.4f}")
print(f"  Calinski-Harabasz:      {km_chi:.2f}")
print(f"  Phân phối clusters:     {dict(pd.Series(km_labels).value_counts().sort_index())}")

# 4.3 PCA để trực quan hóa
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
print(f"\nPCA variance explained: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

palette = plt.cm.Set1(np.linspace(0, 0.8, best_k))
fig, ax = plt.subplots(figsize=(9, 6))
for i in range(best_k):
    mask = km_labels == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[palette[i]], alpha=0.3, s=5,
               label=f'Cluster {i} (n={mask.sum()})')
ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
       ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
       title=f"K-Means (k={best_k}) – PCA Visualization")
ax.legend(markerscale=4); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig("charts/05_kmeans_pca.png")
plt.close(); print("✓ Saved: charts/05_kmeans_pca.png")

# 4.4 Cluster Profile
profile_cols = ['price','number_of_reviews','reviews_per_month',
                'availability_365','minimum_nights','calculated_host_listings_count']
profile_cols = [c for c in profile_cols if c in pre.columns]
prof_df = pre[profile_cols].copy()
prof_df['cluster'] = km_labels
cluster_means = prof_df.groupby('cluster')[profile_cols].mean()
print("\nCluster means (original data):")
print(cluster_means.to_string())

# Radar chart
cp_norm = cluster_means.copy()
for c in profile_cols:
    r = cp_norm[c].max() - cp_norm[c].min()
    if r > 0: cp_norm[c] = (cp_norm[c] - cp_norm[c].min()) / r

N = len(profile_cols)
angles = [n/N*2*np.pi for n in range(N)] + [0]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
colors_r = ['#e74c3c','#3498db','#2ecc71','#f39c12']
for i, row in cp_norm.iterrows():
    vals = row.tolist() + [row.iloc[0]]
    ax.plot(angles, vals, 'o-', lw=2, color=colors_r[i%4], label=f'Cluster {i}')
    ax.fill(angles, vals, alpha=0.1, color=colors_r[i%4])
ax.set_xticks(angles[:-1]); ax.set_xticklabels(profile_cols, fontsize=9)
ax.set_title('Đặc trưng từng cụm K-Means', fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout(); plt.savefig("charts/06_radar_chart.png")
plt.close(); print("✓ Saved: charts/06_radar_chart.png")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 5: DBSCAN CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("5. DBSCAN CLUSTERING")
print("="*60)

# Dùng subsample nhỏ hơn cho DBSCAN (tránh memory explosion)
n_db = 2000
np.random.seed(42)
idx_db = np.random.choice(len(X), n_db, replace=False)
X_db = X[idx_db]
X_pca_db = X_pca[idx_db]

# 5.1 k-Distance Graph để chọn epsilon
print("Tính k-Distance graph...")
nbrs = NearestNeighbors(n_neighbors=5).fit(X_db)
dists, _ = nbrs.kneighbors(X_db)
k_dist = np.sort(dists[:, -1])[::-1]  # Sắp xếp giảm dần

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(k_dist, color='navy', lw=1.5)
for ep, col, ls in [(1.0,'red','--'), (2.0,'orange','--'), (4.0,'green','--')]:
    ax.axhline(ep, color=col, ls=ls, lw=1.5, label=f'ε={ep}')
ax.set(xlabel='Điểm (sắp xếp)', ylabel='5-NN distance',
       title='k-Distance Graph (k=5) – Xác định ε cho DBSCAN',
       ylim=(0, min(10, k_dist[50]*2)))
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("charts/07_kdistance.png")
plt.close(); print("✓ Saved: charts/07_kdistance.png")

# 5.2 Phân tích độ nhạy tham số DBSCAN
print("\nPhân tích độ nhạy DBSCAN (eps, min_samples=5):")
eps_candidates = [1.0, 1.5, 2.0, 3.0, 4.0]
db_results = []

for eps in eps_candidates:
    db = DBSCAN(eps=eps, min_samples=5, algorithm='ball_tree')
    lbl = db.fit_predict(X_db)
    nc = len(set(lbl)) - (1 if -1 in lbl else 0)
    nn = (lbl == -1).sum()
    non_noise = lbl != -1
    if nc >= 2 and non_noise.sum() > nc + 2:
        sil = silhouette_score(X_db[non_noise], lbl[non_noise])
        dbi = davies_bouldin_score(X_db[non_noise], lbl[non_noise])
        chi = calinski_harabasz_score(X_db[non_noise], lbl[non_noise])
    else:
        sil, dbi, chi = -1.0, -1.0, -1.0
    db_results.append({'eps':eps,'n_clusters':nc,'n_noise':nn,'sil':sil,'dbi':dbi,'chi':chi,'labels':lbl})
    print(f"  eps={eps}: {nc} clusters, {nn} noise ({nn/n_db*100:.1f}%), Sil={sil:.4f}")

# Chọn eps tốt nhất
candidates = [r for r in db_results if 2 <= r['n_clusters'] <= 20 and r['sil'] > 0]
best_db = max(candidates, key=lambda r: r['sil']) if candidates else db_results[3]
db_eps = best_db['eps']
db_lbl = best_db['labels']
print(f"\n→ epsilon tối ưu: {db_eps}, {best_db['n_clusters']} clusters, {best_db['n_noise']} noise")

# Chỉ số đánh giá DBSCAN
db_sil = best_db['sil']
db_dbi = best_db['dbi']
db_chi = best_db['chi']
print(f"  Silhouette Score:     {db_sil:.4f}")
print(f"  Davies-Bouldin:       {db_dbi:.4f}")
print(f"  Calinski-Harabasz:    {db_chi:.2f}")

# 5.3 Trực quan hóa DBSCAN
unique_lbl = sorted(set(db_lbl))
cmap = plt.cm.tab10(np.linspace(0, 1, max(len(unique_lbl), 2)))
fig, ax = plt.subplots(figsize=(9, 6))
for i, lbl in enumerate(unique_lbl):
    m = db_lbl == lbl
    if m.sum() == 0: continue
    c = 'lightgray' if lbl == -1 else cmap[i % len(cmap)]
    name = f'Noise (n={m.sum()})' if lbl == -1 else f'Cluster {lbl} (n={m.sum()})'
    ax.scatter(X_pca_db[m, 0], X_pca_db[m, 1], c=[c],
               alpha=0.2 if lbl == -1 else 0.6, s=6, label=name)
ax.set(xlabel='PC1', ylabel='PC2', title=f'DBSCAN (ε={db_eps}) – PCA Visualization')
ax.legend(markerscale=2, fontsize=9); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig("charts/08_dbscan_pca.png")
plt.close(); print("✓ Saved: charts/08_dbscan_pca.png")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 6: SO SÁNH VÀ ĐÁNH GIÁ
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("6. SO SÁNH VÀ ĐÁNH GIÁ")
print("="*60)

print(f"\n{'Chỉ số':<30} {'K-Means':>12} {'DBSCAN':>12}")
print("-" * 55)
print(f"{'Silhouette Score':<30} {km_sil:>12.4f} {db_sil:>12.4f}")
print(f"{'Davies-Bouldin Index':<30} {km_dbi:>12.4f} {db_dbi:>12.4f}")
print(f"{'Calinski-Harabasz':<30} {km_chi:>12.2f} {db_chi:>12.2f}")
print(f"{'Số cụm':<30} {best_k:>12} {best_db['n_clusters']:>12}")
print(f"{'Noise points':<30} {'0':>12} {best_db['n_noise']:>12}")

# Biểu đồ so sánh
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for i, (met, kv, dv) in enumerate([
    ('Silhouette Score', km_sil, db_sil),
    ('Davies-Bouldin (thap tot)', km_dbi, db_dbi),
    ('Calinski-Harabasz (cao tot)', km_chi, db_chi)
]):
    bars = axes[i].bar(['K-Means','DBSCAN'], [max(kv,0), max(dv,0)],
                        color=['steelblue','coral'], edgecolor='k', width=0.5)
    axes[i].set_title(met, fontweight='bold'); axes[i].grid(alpha=0.3, axis='y')
    for bar, v in zip(bars, [kv, dv]):
        axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                     f'{v:.3f}' if v > 0 else 'N/A', ha='center', fontsize=10)
plt.suptitle('So sánh K-Means vs DBSCAN', fontweight='bold')
plt.tight_layout(); plt.savefig("charts/09_comparison.png")
plt.close(); print("✓ Saved: charts/09_comparison.png")

print("\n" + "="*60)
print("HOÀN THÀNH! Tất cả biểu đồ đã lưu vào thư mục charts/")
print("="*60)
