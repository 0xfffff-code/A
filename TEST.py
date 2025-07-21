import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("超市客户K-means聚类分析项目")
print("=" * 60)

# ==================== 数据层面 ====================
print("\n1. 数据层面 - 数据加载与预处理")
print("-" * 40)

# 读取数据
data = pd.read_csv('Mall_Customers.csv')
print(f"数据集形状: {data.shape}")
print(f"数据集列名: {list(data.columns)}")

# 数据基本信息
print("\n数据基本信息:")
print(data.info())

print("\n数据描述性统计:")
print(data.describe())

print("\n缺失值检查:")
print(data.isnull().sum())

# 数据可视化 - 了解数据分布
plt.figure(figsize=(15, 10))

# 年龄分布
plt.subplot(2, 3, 1)
plt.hist(data['Age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('年龄分布')
plt.xlabel('年龄')
plt.ylabel('频数')

# 年收入分布
plt.subplot(2, 3, 2)
plt.hist(data['Annual Income (k$)'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('年收入分布')
plt.xlabel('年收入 (k$)')
plt.ylabel('频数')

# 消费评分分布
plt.subplot(2, 3, 3)
plt.hist(data['Spending Score (1-100)'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
plt.title('消费评分分布')
plt.xlabel('消费评分')
plt.ylabel('频数')

# 性别分布
plt.subplot(2, 3, 4)
gender_counts = data['Gender'].value_counts()
plt.bar(gender_counts.index, gender_counts.values, color=['pink', 'lightblue'])
plt.title('性别分布')
plt.xlabel('性别')
plt.ylabel('人数')

# 年收入 vs 消费评分散点图
plt.subplot(2, 3, 5)
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], alpha=0.6)
plt.title('年收入 vs 消费评分')
plt.xlabel('年收入 (k$)')
plt.ylabel('消费评分')

# 年龄 vs 消费评分散点图
plt.subplot(2, 3, 6)
plt.scatter(data['Age'], data['Spending Score (1-100)'], alpha=0.6)
plt.title('年龄 vs 消费评分')
plt.xlabel('年龄')
plt.ylabel('消费评分')

plt.tight_layout()
plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 方法层面 ====================
print("\n2. 方法层面 - 特征工程与模型构建")
print("-" * 40)

# 选择用于聚类的特征
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

print(f"选择的特征: {features}")
print(f"特征数据形状: {X.shape}")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("数据已标准化")

# 确定最佳聚类数 - 肘部法则
print("\n使用肘部法则确定最佳聚类数...")
inertias = []
silhouette_scores = []
calinski_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    
    # 计算轮廓系数
    if k > 1:
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        calinski_scores.append(calinski_harabasz_score(X_scaled, kmeans.labels_))
    else:
        silhouette_scores.append(0)
        calinski_scores.append(0)

# 绘制肘部图
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('聚类数 (K)')
plt.ylabel('惯性 (Inertia)')
plt.title('肘部法则')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('聚类数 (K)')
plt.ylabel('轮廓系数')
plt.title('轮廓系数')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(K_range, calinski_scores, 'go-')
plt.xlabel('聚类数 (K)')
plt.ylabel('Calinski-Harabasz指数')
plt.title('Calinski-Harabasz指数')
plt.grid(True)

plt.tight_layout()
plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

# 选择最佳聚类数
best_k = K_range[np.argmax(silhouette_scores)]
print(f"基于轮廓系数，最佳聚类数为: {best_k}")

# 构建最终模型
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
data['Cluster'] = final_kmeans.fit_predict(X_scaled)

print(f"\nK-means模型训练完成，聚类数: {best_k}")
print(f"最终惯性: {final_kmeans.inertia_:.2f}")
print(f"轮廓系数: {silhouette_score(X_scaled, final_kmeans.labels_):.3f}")

# ==================== 分析层面 ====================
print("\n3. 分析层面 - 聚类结果分析与验证")
print("-" * 40)

# 聚类结果统计
print("各聚类客户数量:")
cluster_counts = data['Cluster'].value_counts().sort_index()
print(cluster_counts)

print("\n各聚类特征均值:")
cluster_means = data.groupby('Cluster')[features].mean()
print(cluster_means)

# 聚类结果可视化
plt.figure(figsize=(20, 15))

# 3D散点图
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(2, 3, 1, projection='3d')
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i in range(best_k):
    cluster_data = data[data['Cluster'] == i]
    ax.scatter(cluster_data['Age'], cluster_data['Annual Income (k$)'], 
               cluster_data['Spending Score (1-100)'], 
               c=colors[i], label=f'聚类 {i}', alpha=0.7)
ax.set_xlabel('年龄')
ax.set_ylabel('年收入 (k$)')
ax.set_zlabel('消费评分')
ax.set_title('3D聚类结果')
ax.legend()

# 年收入 vs 消费评分
plt.subplot(2, 3, 2)
for i in range(best_k):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], 
                c=colors[i], label=f'聚类 {i}', alpha=0.7)
plt.xlabel('年收入 (k$)')
plt.ylabel('消费评分')
plt.title('年收入 vs 消费评分聚类结果')
plt.legend()

# 年龄 vs 消费评分
plt.subplot(2, 3, 3)
for i in range(best_k):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Age'], cluster_data['Spending Score (1-100)'], 
                c=colors[i], label=f'聚类 {i}', alpha=0.7)
plt.xlabel('年龄')
plt.ylabel('消费评分')
plt.title('年龄 vs 消费评分聚类结果')
plt.legend()

# 年龄 vs 年收入
plt.subplot(2, 3, 4)
for i in range(best_k):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Age'], cluster_data['Annual Income (k$)'], 
                c=colors[i], label=f'聚类 {i}', alpha=0.7)
plt.xlabel('年龄')
plt.ylabel('年收入 (k$)')
plt.title('年龄 vs 年收入聚类结果')
plt.legend()

# 聚类特征箱线图
plt.subplot(2, 3, 5)
data.boxplot(column='Annual Income (k$)', by='Cluster', ax=plt.gca())
plt.title('各聚类年收入分布')
plt.suptitle('')

plt.subplot(2, 3, 6)
data.boxplot(column='Spending Score (1-100)', by='Cluster', ax=plt.gca())
plt.title('各聚类消费评分分布')
plt.suptitle('')

plt.tight_layout()
plt.savefig('clustering_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 聚类特征热力图 - 使用matplotlib替代seaborn
plt.figure(figsize=(10, 8))
cluster_summary = data.groupby('Cluster')[features + ['Gender']].agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean', 
    'Spending Score (1-100)': 'mean',
    'Gender': lambda x: (x == 'Female').mean()  # 女性比例
}).round(2)

cluster_summary.columns = ['平均年龄', '平均年收入', '平均消费评分', '女性比例']

# 创建热力图
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cluster_summary.T.values, cmap='YlOrRd')

# 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('数值', rotation=-90, va="bottom")

# 设置刻度标签
ax.set_xticks(np.arange(len(cluster_summary.index)))
ax.set_yticks(np.arange(len(cluster_summary.columns)))
ax.set_xticklabels([f'聚类{i}' for i in cluster_summary.index])
ax.set_yticklabels(cluster_summary.columns)

# 旋转x轴标签
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# 添加数值标签
for i in range(len(cluster_summary.columns)):
    for j in range(len(cluster_summary.index)):
        text = ax.text(j, i, f'{cluster_summary.iloc[j, i]:.2f}',
                      ha="center", va="center", color="black")

ax.set_title('聚类特征热力图')
plt.tight_layout()
plt.savefig('cluster_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 客户画像分析
print("\n客户画像分析:")
for i in range(best_k):
    cluster_data = data[data['Cluster'] == i]
    print(f"\n聚类 {i} 客户画像:")
    print(f"  客户数量: {len(cluster_data)} ({len(cluster_data)/len(data)*100:.1f}%)")
    print(f"  平均年龄: {cluster_data['Age'].mean():.1f}岁")
    print(f"  平均年收入: {cluster_data['Annual Income (k$)'].mean():.1f}k$")
    print(f"  平均消费评分: {cluster_data['Spending Score (1-100)'].mean():.1f}")
    print(f"  性别分布: 男性{len(cluster_data[cluster_data['Gender']=='Male'])}人, "
          f"女性{len(cluster_data[cluster_data['Gender']=='Female'])}人")

# 模型验证
print("\n4. 模型验证")
print("-" * 40)

# 使用PCA降维可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for i in range(best_k):
    cluster_mask = final_kmeans.labels_ == i
    plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], 
                c=colors[i], label=f'聚类 {i}', alpha=0.7)
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.title('PCA降维后的聚类结果')
plt.legend()

# 聚类稳定性验证 - 多次运行
plt.subplot(1, 2, 2)
stability_scores = []
for _ in range(10):
    kmeans_temp = KMeans(n_clusters=best_k, random_state=None, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    stability_scores.append(silhouette_score(X_scaled, labels_temp))

plt.hist(stability_scores, bins=10, alpha=0.7, color='lightblue', edgecolor='black')
plt.axvline(np.mean(stability_scores), color='red', linestyle='--', label=f'平均值: {np.mean(stability_scores):.3f}')
plt.xlabel('轮廓系数')
plt.ylabel('频数')
plt.title('模型稳定性验证')
plt.legend()

plt.tight_layout()
plt.savefig('model_validation.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"模型稳定性验证 - 10次运行轮廓系数:")
print(f"  平均值: {np.mean(stability_scores):.3f}")
print(f"  标准差: {np.std(stability_scores):.3f}")
print(f"  最小值: {np.min(stability_scores):.3f}")
print(f"  最大值: {np.max(stability_scores):.3f}")

# 保存结果
data.to_csv('clustered_customers.csv', index=False)
print(f"\n聚类结果已保存到 'clustered_customers.csv'")

print("\n" + "=" * 60)
print("K-means聚类分析完成！")
print("=" * 60)
