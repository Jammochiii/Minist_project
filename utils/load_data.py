import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def load_data(use_subset=True, subset_per_class=1000, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

    if use_subset:
        train_indices = []
        for class_idx in range(10):
            class_indices = (np.array(train_dataset.targets) == class_idx).nonzero()[0]
            train_indices.extend(class_indices[:subset_per_class])
        train_dataset = Subset(train_dataset, train_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    print(f"[INFO] 训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")
    return train_loader, test_loader