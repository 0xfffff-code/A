import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class DeepClusteringModel:
    def __init__(self, input_dim, n_clusters, encoding_dim=10):
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.encoding_dim = encoding_dim
        self.autoencoder = None
        self.clustering_layer = None
        self.model = None
        
    def build_autoencoder(self):
        """构建自编码器"""
        # 编码器
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoded')(encoded)
        
        # 解码器
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(decoded)
        
        self.autoencoder = Model(input_layer, decoded)
        return self.autoencoder
    
    def build_clustering_model(self):
        """构建聚类模型"""
        # 聚类层
        clustering_layer = layers.Dense(self.n_clusters, 
                                       activation='softmax', 
                                       name='clustering')(self.autoencoder.get_layer('encoded').output)
        
        self.model = Model(self.autoencoder.input, clustering_layer)
        return self.model
    
    def pretrain_autoencoder(self, X, epochs=50, batch_size=32):
        """预训练自编码器"""
        self.autoencoder.compile(optimizer='adam', loss='mse')
        history = self.autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0)
        return history
    
    def initialize_cluster_centers(self, X):
        """初始化聚类中心"""
        # 使用编码后的特征进行K-means初始化
        encoder = Model(self.autoencoder.input, self.autoencoder.get_layer('encoded').output)
        encoded_X = encoder.predict(X)
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        cluster_centers = kmeans.fit(encoded_X).cluster_centers_
        
        # 设置聚类层的权重
        self.model.get_layer('clustering').set_weights([cluster_centers.T, np.zeros(self.n_clusters)])
        return cluster_centers
    
    def train_clustering(self, X, epochs=100, batch_size=32, update_interval=10):
        """训练聚类模型"""
        # 自定义损失函数
        def clustering_loss(y_true, y_pred):
            # 计算聚类损失
            q = y_pred
            p = tf.square(q) / tf.reduce_sum(q, axis=0)
            p = p / tf.reduce_sum(p, axis=1, keepdims=True)
            
            # KL散度损失
            kl_loss = tf.reduce_sum(p * tf.math.log(tf.clip_by_value(p / tf.clip_by_value(q, 1e-8, 1.0), 1e-8, 1e8)))
            return kl_loss
        
        self.model.compile(optimizer='adam', loss=clustering_loss)
        
        # 训练过程
        for epoch in range(epochs):
            if epoch % update_interval == 0:
                # 更新目标分布
                q = self.model.predict(X)
                p = np.square(q) / np.sum(q, axis=0)
                p = p / np.sum(p, axis=1, keepdims=True)
                
                # 设置目标分布
                y = p
            
            # 训练一个epoch
            self.model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0)
        
        return self.model
    
    def predict_clusters(self, X):
        """预测聚类标签"""
        q = self.model.predict(X)
        return np.argmax(q, axis=1)

def main():
    print("=" * 60)
    print("深度学习聚类算法")
    print("=" * 60)
    
    # 加载数据
    data = pd.read_csv('Mall_Customers.csv')
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = data[features]
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 创建深度学习聚类模型
    n_clusters = 5
    deep_model = DeepClusteringModel(input_dim=X_scaled.shape[1], 
                                   n_clusters=n_clusters, 
                                   encoding_dim=10)
    
    print("1. 构建自编码器...")
    deep_model.build_autoencoder()
    
    print("2. 预训练自编码器...")
    history = deep_model.pretrain_autoencoder(X_scaled, epochs=50)
    
    print("3. 构建聚类模型...")
    deep_model.build_clustering_model()
    
    print("4. 初始化聚类中心...")
    deep_model.initialize_cluster_centers(X_scaled)
    
    print("5. 训练聚类模型...")
    deep_model.train_clustering(X_scaled, epochs=100)
    
    print("6. 预测聚类结果...")
    cluster_labels = deep_model.predict_clusters(X_scaled)
    
    # 评估结果
    silhouette = silhouette_score(X_scaled, cluster_labels)
    print(f"\n深度学习聚类结果:")
    print(f"轮廓系数: {silhouette:.3f}")
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 原始数据分布
    plt.subplot(1, 3, 1)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.6)
    plt.title('原始数据分布')
    plt.xlabel('标准化年龄')
    plt.ylabel('标准化收入')
    
    # 聚类结果
    plt.subplot(1, 3, 2)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(n_clusters):
        mask = cluster_labels == i
        plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
                   c=colors[i], label=f'聚类 {i}', alpha=0.7)
    plt.title('深度学习聚类结果')
    plt.xlabel('标准化年龄')
    plt.ylabel('标准化收入')
    plt.legend()
    
    # 聚类分布
    plt.subplot(1, 3, 3)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    plt.bar(unique, counts, color=colors[:len(unique)])
    plt.title('聚类分布')
    plt.xlabel('聚类编号')
    plt.ylabel('客户数量')
    
    plt.tight_layout()
    plt.savefig('deep_clustering_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("深度学习聚类分析完成！")

if __name__ == '__main__':
    main() 