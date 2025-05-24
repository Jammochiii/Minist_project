import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 特征可视化
def visualize_features(model, test_loader, max_samples_per_class=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    class_samples = {i: [] for i in range(10)}
    collected_classes = {i: 0 for i in range(10)}

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            for i in range(len(target)):
                label = target[i].item()
                if collected_classes[label] < max_samples_per_class:
                    class_samples[label].append({
                        'conv2': model.conv_features['conv2'][i].cpu().numpy(),
                        'fc2': model.fc_features['fc2'][i].cpu().numpy(),
                        'fc4': model.fc_features['fc4'][i].cpu().numpy()
                    })
                    collected_classes[label] += 1
            if all(count >= max_samples_per_class for count in collected_classes.values()):
                break

    all_features = {'conv2': [], 'fc2': [], 'fc4': []}
    all_labels = []
    for label in range(10):
        for sample in class_samples[label]:
            all_features['conv2'].append(sample['conv2'].reshape(-1))
            all_features['fc2'].append(sample['fc2'])
            all_features['fc4'].append(sample['fc4'])
            all_labels.append(label)
    for layer in all_features:
        all_features[layer] = np.array(all_features[layer])
    all_labels = np.array(all_labels)
    print(f"[INFO] 特征可视化采样共 {len(all_labels)} 张，每类 {max_samples_per_class} 张")

    visualize_pca(all_features, all_labels)
    visualize_tsne(all_features, all_labels)

def visualize_pca(features, labels):
    plt.figure(figsize=(18, 5))
    for i, (layer_name, layer_features) in enumerate(features.items()):
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(layer_features)
        plt.subplot(1, 3, i+1)
        for label in range(10):
            indices = labels == label
            plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1],
                        label=str(label), alpha=0.7, s=30)
        plt.title(f'PCA Visualization ({layer_name})')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(title='Digits')
    plt.tight_layout()
    plt.savefig('pca_visualization.png')
    plt.close()

def visualize_tsne(features, labels):
    plt.figure(figsize=(18, 5))
    for i, (layer_name, layer_features) in enumerate(features.items()):
        if layer_features.shape[0] > 1000:
            indices = np.random.choice(layer_features.shape[0], 1000, replace=False)
            sample_features = layer_features[indices]
            sample_labels = labels[indices]
        else:
            sample_features = layer_features
            sample_labels = labels
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(sample_features)
        plt.subplot(1, 3, i+1)
        for label in range(10):
            indices = sample_labels == label
            plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1],
                        label=str(label), alpha=0.7, s=30)
        plt.title(f't-SNE Visualization ({layer_name})')
        plt.xlabel('t-SNE1')
        plt.ylabel('t-SNE2')
        plt.legend(title='Digits')
    plt.tight_layout()
    plt.savefig('tsne_visualization.png')
    plt.close()