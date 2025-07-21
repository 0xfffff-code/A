import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("高级聚类算法对比分析")
print("=" * 60)

# 读取数据
data = pd.read_csv('Mall_Customers.csv')
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定义聚类算法
algorithms = {
    'K-Means': KMeans(n_clusters=5, random_state=42, n_init=10),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Hierarchical': AgglomerativeClustering(n_clusters=5),
    'Spectral': SpectralClustering(n_clusters=5, random_state=42),
    'Gaussian Mixture': GaussianMixture(n_components=5, random_state=42)
}

# 评估指标
metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']

# 存储结果
results = {}

print("\n1. 多算法聚类对比")
print("-" * 40)

for name, algorithm in algorithms.items():
    print(f"\n运行 {name} 算法...")
    
    # 训练模型
    if name == 'Gaussian Mixture':
        labels = algorithm.fit_predict(X_scaled)
    else:
        labels = algorithm.fit_predict(X_scaled)
    
    # 计算评估指标
    n_clusters = len(np.unique(labels[labels != -1]))  # 排除噪声点
    
    if n_clusters > 1:
        silhouette = silhouette_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        davies = davies_bouldin_score(X_scaled, labels)
    else:
        silhouette = calinski = davies = 0
    
    results[name] = {
        'labels': labels,
        'n_clusters': n_clusters,
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'davies_bouldin': davies
    }
    
    print(f"  聚类数: {n_clusters}")
    print(f"  轮廓系数: {silhouette:.3f}")
    print(f"  Calinski-Harabasz指数: {calinski:.3f}")
    print(f"  Davies-Bouldin指数: {davies:.3f}")

# 可视化对比结果 - 分为两个图
# 第一个图：算法性能对比
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
algo_names = list(results.keys())
silhouette_scores = [results[name]['silhouette'] for name in algo_names]
plt.bar(algo_names, silhouette_scores, color='skyblue')
plt.title('轮廓系数对比')
plt.ylabel('轮廓系数')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
calinski_scores = [results[name]['calinski_harabasz'] for name in algo_names]
plt.bar(algo_names, calinski_scores, color='lightgreen')
plt.title('Calinski-Harabasz指数对比')
plt.ylabel('指数值')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
davies_scores = [results[name]['davies_bouldin'] for name in algo_names]
plt.bar(algo_names, davies_scores, color='lightcoral')
plt.title('Davies-Bouldin指数对比')
plt.ylabel('指数值')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('algorithm_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 第二个图：聚类结果可视化
plt.figure(figsize=(20, 8))

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# 使用t-SNE降维进行可视化
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

for i, (name, result) in enumerate(results.items()):
    plt.subplot(2, 3, i+1)  # 2行3列布局，最多6个算法
    labels = result['labels']
    
    # 为每个聚类分配颜色
    unique_labels = np.unique(labels)
    for j, label in enumerate(unique_labels):
        if label == -1:  # 噪声点
            mask = labels == label
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c='black', marker='x', s=20, alpha=0.6, label='噪声')
        else:
            mask = labels == label
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=colors[j % len(colors)], alpha=0.7, label=f'聚类{label}')
    
    plt.title(f'{name} 聚类结果')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    if i == 0:  # 只在第一个子图显示图例
        plt.legend()

plt.tight_layout()
plt.savefig('algorithm_clustering_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n2. 创新特征工程")
print("-" * 40)

# 创建新特征
data['Income_Spending_Ratio'] = data['Annual Income (k$)'] / (data['Spending Score (1-100)'] + 1)
data['Age_Income_Ratio'] = data['Age'] / (data['Annual Income (k$)'] + 1)
data['Spending_Efficiency'] = data['Spending Score (1-100)'] / (data['Age'] + 1)

# 创建客户价值评分
data['Customer_Value'] = (data['Annual Income (k$)'] * 0.4 + 
                         data['Spending Score (1-100)'] * 0.4 + 
                         (100 - data['Age']) * 0.2)

print("新增特征:")
print("- Income_Spending_Ratio: 收入消费比")
print("- Age_Income_Ratio: 年龄收入比")
print("- Spending_Efficiency: 消费效率")
print("- Customer_Value: 客户价值评分")

# 使用新特征进行聚类
new_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 
                'Income_Spending_Ratio', 'Customer_Value']
X_new = data[new_features]

scaler_new = StandardScaler()
X_new_scaled = scaler_new.fit_transform(X_new)

# 对比原始特征和新特征
kmeans_original = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_new = KMeans(n_clusters=5, random_state=42, n_init=10)

labels_original = kmeans_original.fit_predict(X_scaled)
labels_new = kmeans_new.fit_predict(X_new_scaled)

print(f"\n特征工程效果对比:")
print(f"原始特征轮廓系数: {silhouette_score(X_scaled, labels_original):.3f}")
print(f"新特征轮廓系数: {silhouette_score(X_new_scaled, labels_new):.3f}")

print("\n3. 动态聚类数优化")
print("-" * 40)

# 自适应聚类数选择
def find_optimal_clusters(X, max_k=10):
    """使用多种指标自动选择最佳聚类数"""
    scores = {}
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        if len(np.unique(labels)) > 1:
            scores[k] = {
                'silhouette': silhouette_score(X, labels),
                'calinski': calinski_harabasz_score(X, labels),
                'davies': davies_bouldin_score(X, labels),
                'inertia': kmeans.inertia_
            }
    
    # 综合评分
    for k in scores:
        # 归一化各指标
        silhouette_norm = scores[k]['silhouette'] / max(s['silhouette'] for s in scores.values())
        calinski_norm = scores[k]['calinski'] / max(s['calinski'] for s in scores.values())
        davies_norm = min(s['davies'] for s in scores.values()) / scores[k]['davies']  # 越小越好
        inertia_norm = min(s['inertia'] for s in scores.values()) / scores[k]['inertia']  # 越小越好
        
        # 综合评分
        scores[k]['composite_score'] = (silhouette_norm + calinski_norm + davies_norm + inertia_norm) / 4
    
    return scores

scores = find_optimal_clusters(X_new_scaled)
optimal_k = max(scores.keys(), key=lambda k: scores[k]['composite_score'])

print(f"自适应选择的最佳聚类数: {optimal_k}")
print(f"综合评分: {scores[optimal_k]['composite_score']:.3f}")

# 可视化优化过程
plt.figure(figsize=(15, 5))

k_values = list(scores.keys())
composite_scores = [scores[k]['composite_score'] for k in k_values]

plt.subplot(1, 3, 1)
plt.plot(k_values, composite_scores, 'bo-')
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'最佳K={optimal_k}')
plt.xlabel('聚类数 (K)')
plt.ylabel('综合评分')
plt.title('自适应聚类数选择')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
silhouette_scores = [scores[k]['silhouette'] for k in k_values]
plt.plot(k_values, silhouette_scores, 'ro-')
plt.xlabel('聚类数 (K)')
plt.ylabel('轮廓系数')
plt.title('轮廓系数变化')
plt.grid(True)

plt.subplot(1, 3, 3)
inertia_scores = [scores[k]['inertia'] for k in k_values]
plt.plot(k_values, inertia_scores, 'go-')
plt.xlabel('聚类数 (K)')
plt.ylabel('惯性')
plt.title('惯性变化')
plt.grid(True)

plt.tight_layout()
plt.savefig('adaptive_clustering.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n4. 客户生命周期分析")
print("-" * 40)

# 基于年龄和消费行为分析客户生命周期
data['Life_Stage'] = pd.cut(data['Age'], 
                           bins=[0, 25, 35, 50, 65, 100], 
                           labels=['青年期', '成长期', '成熟期', '稳定期', '退休期'])

# 分析不同生命阶段的客户特征
life_stage_analysis = data.groupby('Life_Stage').agg({
    'Annual Income (k$)': ['mean', 'std'],
    'Spending Score (1-100)': ['mean', 'std'],
    'Customer_Value': ['mean', 'std']
}).round(2)

print("客户生命周期分析:")
print(life_stage_analysis)

# 可视化生命周期分析
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
life_stage_income = data.groupby('Life_Stage')['Annual Income (k$)'].mean()
plt.bar(life_stage_income.index, life_stage_income.values, color='lightblue')
plt.title('各生命阶段平均收入')
plt.ylabel('平均收入 (k$)')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
life_stage_spending = data.groupby('Life_Stage')['Spending Score (1-100)'].mean()
plt.bar(life_stage_spending.index, life_stage_spending.values, color='lightgreen')
plt.title('各生命阶段平均消费评分')
plt.ylabel('平均消费评分')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
life_stage_value = data.groupby('Life_Stage')['Customer_Value'].mean()
plt.bar(life_stage_value.index, life_stage_value.values, color='lightcoral')
plt.title('各生命阶段客户价值')
plt.ylabel('平均客户价值')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('lifecycle_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n5. 营销策略推荐系统")
print("-" * 40)

# 基于聚类结果生成营销策略
def generate_marketing_strategy(cluster_data, cluster_id):
    """为每个聚类生成个性化营销策略"""
    avg_age = cluster_data['Age'].mean()
    avg_income = cluster_data['Annual Income (k$)'].mean()
    avg_spending = cluster_data['Spending Score (1-100)'].mean()
    avg_value = cluster_data['Customer_Value'].mean()
    
    strategies = []
    
    # 基于年龄的策略
    if avg_age < 30:
        strategies.append("社交媒体营销，年轻化产品推广")
    elif avg_age < 50:
        strategies.append("家庭套餐，中高端产品推荐")
    else:
        strategies.append("健康产品，舒适服务体验")
    
    # 基于收入的策略
    if avg_income > 80:
        strategies.append("VIP会员服务，奢侈品推荐")
    elif avg_income > 50:
        strategies.append("中高端产品，会员积分优惠")
    else:
        strategies.append("性价比产品，促销活动")
    
    # 基于消费评分的策略
    if avg_spending > 70:
        strategies.append("新品优先体验，专属客服")
    elif avg_spending > 40:
        strategies.append("个性化推荐，消费引导")
    else:
        strategies.append("基础服务，价格敏感型营销")
    
    # 基于客户价值的策略
    if avg_value > 60:
        strategies.append("高价值客户维护，专属权益")
    elif avg_value > 40:
        strategies.append("价值提升计划，交叉销售")
    else:
        strategies.append("基础客户服务，价值挖掘")
    
    return strategies

# 使用最佳聚类结果生成策略
best_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data['Best_Cluster'] = best_kmeans.fit_predict(X_new_scaled)

print("个性化营销策略推荐:")
for i in range(optimal_k):
    cluster_data = data[data['Best_Cluster'] == i]
    strategies = generate_marketing_strategy(cluster_data, i)
    
    print(f"\n聚类 {i} 营销策略:")
    print(f"  客户数量: {len(cluster_data)}")
    print(f"  平均客户价值: {cluster_data['Customer_Value'].mean():.1f}")
    print("  推荐策略:")
    for j, strategy in enumerate(strategies, 1):
        print(f"    {j}. {strategy}")

# 保存增强结果
data.to_csv('enhanced_clustering_results.csv', index=False)
print(f"\n增强分析结果已保存到 'enhanced_clustering_results.csv'")

print("\n" + "=" * 60)
print("高级聚类分析完成！")
print("=" * 60) 