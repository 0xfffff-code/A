import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CustomerMonitoringSystem:
    def __init__(self, data_file='Mall_Customers.csv'):
        """初始化客户监控系统"""
        self.data_file = data_file
        self.data = pd.read_csv(data_file)
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.cluster_centers = None
        self.anomaly_threshold = 2.0  # 异常检测阈值
        self.alert_history = []
        
        # 初始化聚类模型
        self._initialize_clustering()
    
    def _initialize_clustering(self):
        """初始化聚类模型"""
        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        X = self.data[features]
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练K-means模型
        self.kmeans_model = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.kmeans_model.fit(X_scaled)
        self.cluster_centers = self.kmeans_model.cluster_centers_
        
        print("聚类模型初始化完成")
    
    def detect_anomalies(self, new_data):
        """检测异常客户"""
        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        X_new = new_data[features]
        X_new_scaled = self.scaler.transform(X_new)
        
        # 计算到聚类中心的距离
        distances = []
        for i, point in enumerate(X_new_scaled):
            cluster_label = self.kmeans_model.predict([point])[0]
            center = self.cluster_centers[cluster_label]
            distance = np.linalg.norm(point - center)
            distances.append(distance)
        
        # 标记异常点
        anomalies = []
        for i, distance in enumerate(distances):
            if distance > self.anomaly_threshold:
                anomalies.append({
                    'customer_id': new_data.iloc[i]['CustomerID'],
                    'distance': distance,
                    'cluster': self.kmeans_model.predict([X_new_scaled[i]])[0],
                    'timestamp': datetime.now().isoformat()
                })
        
        return anomalies
    
    def monitor_customer_changes(self, old_data, new_data):
        """监控客户行为变化"""
        changes = []
        
        for _, new_customer in new_data.iterrows():
            customer_id = new_customer['CustomerID']
            old_customer = old_data[old_data['CustomerID'] == customer_id]
            
            if not old_customer.empty:
                old_customer = old_customer.iloc[0]
                
                # 检测关键指标变化
                income_change = new_customer['Annual Income (k$)'] - old_customer['Annual Income (k$)']
                spending_change = new_customer['Spending Score (1-100)'] - old_customer['Spending Score (1-100)']
                
                # 判断是否有显著变化
                if abs(income_change) > 10 or abs(spending_change) > 20:
                    changes.append({
                        'customer_id': customer_id,
                        'income_change': income_change,
                        'spending_change': spending_change,
                        'timestamp': datetime.now().isoformat(),
                        'severity': 'high' if abs(income_change) > 20 or abs(spending_change) > 30 else 'medium'
                    })
        
        return changes
    
    def generate_alerts(self, anomalies, changes):
        """生成预警信息"""
        alerts = []
        
        # 异常客户预警
        for anomaly in anomalies:
            alerts.append({
                'type': 'anomaly',
                'message': f"客户 {anomaly['customer_id']} 行为异常，距离聚类中心 {anomaly['distance']:.2f}",
                'severity': 'high',
                'timestamp': anomaly['timestamp']
            })
        
        # 客户变化预警
        for change in changes:
            if change['severity'] == 'high':
                alerts.append({
                    'type': 'behavior_change',
                    'message': f"客户 {change['customer_id']} 行为发生重大变化",
                    'severity': 'high',
                    'timestamp': change['timestamp']
                })
        
        return alerts
    
    def update_model(self, new_data):
        """更新聚类模型"""
        # 合并新数据
        self.data = pd.concat([self.data, new_data], ignore_index=True)
        
        # 重新训练模型
        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        X = self.data[features]
        X_scaled = self.scaler.fit_transform(X)
        
        self.kmeans_model.fit(X_scaled)
        self.cluster_centers = self.kmeans_model.cluster_centers_
        
        print(f"模型已更新，当前数据量: {len(self.data)}")
    
    def generate_report(self):
        """生成监控报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_customers': len(self.data),
            'total_alerts': len(self.alert_history),
            'recent_alerts': self.alert_history[-10:] if self.alert_history else [],
            'cluster_distribution': self.data['Cluster'].value_counts().to_dict() if 'Cluster' in self.data.columns else {}
        }
        
        return report
    
    def save_alert_history(self, filename='alert_history.json'):
        """保存预警历史"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.alert_history, f, ensure_ascii=False, indent=2)
    
    def load_alert_history(self, filename='alert_history.json'):
        """加载预警历史"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.alert_history = json.load(f)
        except FileNotFoundError:
            self.alert_history = []

# 模拟实时数据流
def simulate_real_time_data(original_data, num_new_customers=10):
    """模拟生成新的客户数据"""
    new_customers = []
    
    for i in range(num_new_customers):
        # 基于原始数据分布生成新客户
        random_customer = original_data.sample(1).iloc[0]
        
        # 添加一些随机变化
        new_customer = {
            'CustomerID': len(original_data) + i + 1,
            'Gender': random_customer['Gender'],
            'Age': random_customer['Age'] + np.random.randint(-5, 6),
            'Annual Income (k$)': random_customer['Annual Income (k$)'] + np.random.randint(-10, 11),
            'Spending Score (1-100)': random_customer['Spending Score (1-100)'] + np.random.randint(-15, 16)
        }
        
        # 确保数值在合理范围内
        new_customer['Age'] = max(18, min(80, new_customer['Age']))
        new_customer['Annual Income (k$)'] = max(15, new_customer['Annual Income (k$)'])
        new_customer['Spending Score (1-100)'] = max(1, min(100, new_customer['Spending Score (1-100)']))
        
        new_customers.append(new_customer)
    
    return pd.DataFrame(new_customers)

# 主监控程序
def main():
    print("=" * 60)
    print("客户行为实时监控系统")
    print("=" * 60)
    
    # 初始化监控系统
    monitor = CustomerMonitoringSystem()
    
    # 模拟多轮监控
    for round_num in range(5):
        print(f"\n=== 第 {round_num + 1} 轮监控 ===")
        
        # 生成新数据
        new_data = simulate_real_time_data(monitor.data, num_new_customers=5)
        print(f"新增客户数: {len(new_data)}")
        
        # 检测异常
        anomalies = monitor.detect_anomalies(new_data)
        print(f"检测到异常客户: {len(anomalies)}")
        
        # 监控变化（如果有历史数据）
        if round_num > 0:
            changes = monitor.monitor_customer_changes(monitor.data, new_data)
            print(f"检测到客户变化: {len(changes)}")
        else:
            changes = []
        
        # 生成预警
        alerts = monitor.generate_alerts(anomalies, changes)
        
        # 记录预警历史
        monitor.alert_history.extend(alerts)
        
        # 显示预警信息
        for alert in alerts:
            print(f"⚠️  {alert['severity'].upper()}: {alert['message']}")
        
        # 更新模型
        monitor.update_model(new_data)
        
        # 等待一段时间
        time.sleep(2)
    
    # 生成最终报告
    report = monitor.generate_report()
    print(f"\n=== 监控报告 ===")
    print(f"总客户数: {report['total_customers']}")
    print(f"总预警数: {report['total_alerts']}")
    print(f"聚类分布: {report['cluster_distribution']}")
    
    # 保存预警历史
    monitor.save_alert_history()
    print("预警历史已保存")

if __name__ == "__main__":
    main() 