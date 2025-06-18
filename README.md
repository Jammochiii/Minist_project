# MNIST Project

本项目是一个基于 PyTorch 实现的 MNIST 手写数字识别系统，包含数据加载、神经网络模型构建、训练与测试流程，以及特征提取和可视化分析。适合学习深度学习基本流程与特征可视化实践。

## 目录结构

```
mnist_project/
│
├── data/                  # 实验数据
├── utils/load_data.py     # 数据加载与预处理
├── model/CustomNet.py     # 神经网络结构
├── train/train.py         # 训练与测试流程
├── feature/feature_vis.py # 特征提取与可视化
├── main.py                # 主入口
├── result                 # 实验结果
```

## 项目功能

- 使用自定义卷积神经网络（CustomNet）对 MNIST 数据集进行分类
- 支持训练损失、准确率的记录与可视化
- 支持对不同网络层输出特征的 PCA/t-SNE 可视化分析
- 支持固定随机种子保证实验复现

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

1. 下载 MNIST 数据集、训练模型并进行特征可视化：

```bash
python main.py
```

2. 训练和可视化过程中会在`reult`中自动生成如下文件：
   - `train_losses.txt` / `test_losses.txt`：每 epoch 损失
   - `train_accuracies.txt` / `test_accuracies.txt`：每 epoch 准确率
   - `training_curves.png`：训练过程曲线
   - `pca_visualization.png` / `tsne_visualization.png`：特征可视化结果

## 主要文件说明

- `main.py`：主入口，完成数据加载、模型构建、训练与特征可视化
- `utils/load_data.py`：加载 MNIST 数据集，可设定每类样本数等参数
- `model/CustomNet.py`：自定义卷积神经网络，含多层卷积与全连接，支持特征钩子
- `train/train.py`：训练与测试的主流程，包括损失与准确率记录、曲线绘制
- `feature/feature_vis.py`：对不同层输出进行降维和可视化

## 结果示例

训练后在 `training_curves.png`、`pca_visualization.png`、`tsne_visualization.png` 可查看效果。
