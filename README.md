Read file: requirements.txt
以下是一份适合本项目的 README 示例：

---

# 超市客户聚类分析与智能营销系统

## 项目简介

本项目以超市客户数据为基础，综合运用多种聚类算法（K-means、DBSCAN、层次聚类、谱聚类、高斯混合模型、深度聚类等），实现客户分群、特征工程、模型评估、生命周期分析、个性化营销策略推荐，并开发了实时客户行为监控与预警系统。项目涵盖数据处理、建模、分析和业务应用的完整流程，适用于零售行业的客户智能管理和精准营销。

## 目录结构

```
.
├── advanced_clustering.py         # 多算法聚类与特征工程
├── deep_clustering.py             # 深度学习聚类
├── real_time_monitoring.py        # 实时客户行为监控与预警
├── TEST.py                        # K-means聚类完整流程与可视化
├── Mall_Customers.csv             # 样例客户数据
├── clustering_results.csv         # K-means聚类结果
├── enhanced_clustering_results.csv# 增强特征聚类结果
├── clustered_customers.csv        # 聚类客户明细
├── alert_history.json             # 预警历史记录
├── 各类分析与可视化图片（.png）
├── requirements.txt               # 依赖包列表
└── ...
```

## 环境依赖

请确保已安装以下Python库（推荐使用Python 3.10及以上版本）：

```
pandas~=2.3.1
numpy~=1.26.4
matplotlib~=3.10.3
scikit-learn~=1.7.1
tensorflow~=2.19.0
streamlit~=1.28.0
plotly~=5.15.0
seaborn~=0.11.0
```

安装依赖：
```bash
pip install -r requirements.txt
```

## 快速开始

1. **准备数据**  
   确保`Mall_Customers.csv`等数据文件在项目根目录下。

2. **运行聚类分析**  
   - 基础K-means聚类与可视化：
     ```bash
     python TEST.py
     ```
   - 多算法聚类与特征工程对比：
     ```bash
     python advanced_clustering.py
     ```
   - 深度学习聚类：
     ```bash
     python deep_clustering.py
     ```
   - 实时客户行为监控与预警模拟：
     ```bash
     python real_time_monitoring.py
     ```

3. **查看结果**  
   - 聚类结果、客户画像、生命周期分析等输出在csv和png文件中。
   - 预警历史保存在`alert_history.json`。

## 主要功能

- 多种聚类算法对比与评估
- 创新特征工程与客户价值建模
- 深度学习聚类方法
- 动态聚类数自适应优化
- 客户生命周期分析
- 个性化营销策略自动推荐
- 实时客户行为监控与预警

## 可视化示例

项目会自动生成多种分析图表（如肘部法则、聚类分布、生命周期分析、模型验证等），便于直观理解聚类效果和客户结构。

## 适用场景

- 零售企业客户分群与精准营销
- 客户价值分析与生命周期管理
- 客户行为实时监控与风险预警
- 数据挖掘与无监督学习教学案例

