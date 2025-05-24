import torch
import numpy as np

from utils.load_data import load_data
from model.CustomNet import CustomNet, count_parameters
from train.train import train_model
from feature.feature_vis import visualize_features

# 设置随机种子确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main():
    set_seed(42)
    print("Loading data...")
    train_loader, test_loader = load_data(use_subset=True)
    print("Creating model...")
    model = CustomNet()
    print(f"[INFO] 网络层数: {model.layer_count}，总参数量: {count_parameters(model)}")
    print("Training model...")
    model = train_model(model, train_loader, test_loader, epochs=10)
    print("Visualizing features...")
    visualize_features(model, test_loader)
    print("Training and visualization completed!")

if __name__ == '__main__':
    main()